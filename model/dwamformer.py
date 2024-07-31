import torch
import torch.nn as nn
import torch.nn.functional as F
from module.dwamformer_layer import DWAMFormerEncoder
from module.utils import create_PositionalEncoding


class AttentionPool(nn.Module):
    """
    Attention pooling is using feature levels to generate attention score for weighted sum.
    """

    def __init__(self, input_dim, reduction=2):
        super(AttentionPool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, input_dim))
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

        self.attention_weights_by_dim = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        b, t, m, c = x.size()
        x = x.transpose(-1, -2)  # b, t, c, m
        y = torch.full((b, t, m), 1 / m, device='cuda')
        y = F.softmax(self.attention_weights(y), dim=-1).unsqueeze(dim=-1)
        x_attn = torch.matmul(x, y).squeeze(dim=-1)
        """

        """
        x = x.transpose(-1, -2)  # b, t, c, m
        y = self.attention_weights(x)
        x_attn = x * y
        x_attn = torch.sum(x_attn, dim=-1)
        """

        b, t, m, c = x.size()
        x = x.transpose(-1, -2).contiguous().view(b, t, m * c)
        y = self.attention_weights(x)

        x = x.view(b, t, c, m)
        y = y.view(b, t, c, m)
        y = F.softmax(y, dim=-1)

        x_attn = x * y
        x_attn = torch.sum(x_attn, dim=-1)

        return x_attn


class MergeBlock(nn.Module):
    """ Merge features between two phases.

        The number of tokens is decreased while the dimension of token is increased.
    """

    def __init__(self,
                 in_channels,  # 512
                 merge_scale: int,  # 5
                 expand: int = 2  # 1
                 ):
        """
        input_dim, # [512, 512, 512]
        _ms, # [5, 5, 4, -1]
        expand=exp # [1, 1, 2]
        """
        super().__init__()

        out_channels = in_channels * expand  # d expand to 512
        self.MS = int(merge_scale)  # 5
        # self.pool = nn.AdaptiveAvgPool2d((1, in_channels))  # 2D average pooling layer with a window size of (1, 512)
        self.attn_pool = AttentionPool(in_channels * self.MS)
        self.fc = nn.Linear(in_channels, out_channels)  # Linear to expand d
        self.norm = nn.LayerNorm(out_channels)  # LayerNorm

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS  # ms = 5

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        """
        Ensure that the length of the input features is divisible by `ms`. If it is not divisible, 
        pad the input features with zeros at the end to make their length divisible by `ms`.
        """

        x = x.view(B, T // ms, ms, C)  # (B, T, C) -> (B, T // ms, ms, C)
        # x = self.pool(x).squeeze(dim=-2)
        x = self.attn_pool(x)
        """
        self.pool(x) -> (B, T // ms, 1, C)
        squeeze(dim=-2) -> (B, T // ms, C)
        
        pool `ms` features into a single feature.
        
        """
        x = self.norm(self.fc(x))
        """
        self.fc(x) -> (B, T // ms, C * expand)
        """

        return x


class DWAMFormerBlock(nn.Module):
    def __init__(self,
                 num_layers,  # 2
                 embed_dim,  # 512
                 ffn_embed_dim=2304,  # ffn 256
                 local_size=0,  # Tw 5
                 num_heads=8,  # 8
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation='relu',
                 use_position=False):
        """
        num, # [2, 2, 4, 4]
        input_dim, # [512, 512, 512, 1024]
        ffn_embed_dim, # [256, 256, 256, 512]
        _l, # [5, 8, 8, -1]
        num_heads, # 8
        dropout, # 0.1
        attention_dropout, # 0.1
        use_position=use_position
        """

        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None  # Generate positional encoding of shape (2000, 512)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.local = int(local_size)
        self.layers = nn.ModuleList([DWAMFormerEncoder(embed_dim,  # 512
                                                       ffn_embed_dim,  # 256
                                                       local_size,  # 5
                                                       num_heads,  # 8
                                                       dropout,  # 0.1
                                                       attention_dropout,  # 0.1
                                                       activation,  # 'relu'
                                                       overlap=True)
                                     for _ in range(num_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                """
                Xavier initialization aims to maintain consistent variance between the inputs and outputs of each layer at the start of training, 
                thereby preventing gradients from vanishing or exploding in deep networks.

                This initialization method sets parameter values so that each parameter is uniformly distributed within a 
                range calculated based on the dimensions of the parameter tensor. It helps keep the variance of the activation 
                function unchanged during both forward and backward propagation.
                """

    def forward(self, x, window):
        output = self.input_norm(x)  # LayerNorm

        for layer in self.layers:
            output = layer(output, window, self.position)  # Input the normalized inputs along with positional encoding into the Encoder.

        return output


class DWAMFormer(nn.Module):
    """
    DWAM-Former is a transformer structure. By three different stages, as frame stage, phoneme stage, and word stage.
    """

    def __init__(self,
                 input_dim,  # Adjust the input feature dimension to be divisible by `num_heads`, specifically 8.
                 ffn_embed_dim,  # MSA ffn embedding（input_dim//2）
                 num_layers,
                 num_heads, 
                 hop,  
                 num_classes,  
                 expand,
                 dropout=0.1, attention_dropout=0.1, device='cuda', _ms=None, **kwargs):

        """
        input_dim = (512 // 8) * 8 = 512
        ffn_embed_dim = 256
        num_layers = [2, 2, 4, 4]
        num_heads = 8
        hop = 0.01
        num_classes = 2
        expand = [1, 1, 2, -1]
        """

        super().__init__()

        self.input_dim = input_dim // num_heads * num_heads  # input_dim = 512

        self.phoneme_length_mlp = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )

        assert isinstance(num_layers, list)

        self.frame_encoder_block = DWAMFormerBlock(3,  # [2, 2, 4, 4]
                                                   self.input_dim,  # [512, 512, 512, 1024]
                                                   ffn_embed_dim,  # [256, 256, 256, 512]
                                                   5,  # [5, 8, 8, -1]
                                                   num_heads,  # 8
                                                   dropout,  # 0.1
                                                   attention_dropout,  # 0.1
                                                   use_position=True)  # DWAMFormerBlock

        self.phoneme_encoder_block = DWAMFormerBlock(3,  # [2, 2, 4, 4]
                                                     self.input_dim,  # [512, 512, 512, 1024]
                                                     ffn_embed_dim,  # [256, 256, 256, 512]
                                                     8,  # [5, 8, 8, -1]
                                                     num_heads,  # 8
                                                     dropout,  # 0.1
                                                     attention_dropout,  # 0.1
                                                     use_position=False)  # DWAMFormerBlock

        self.word_encoder_block = DWAMFormerBlock(6,  # [2, 2, 4, 4]
                                                  self.input_dim,  # [512, 512, 512, 1024]
                                                  ffn_embed_dim,  # [256, 256, 256, 512]
                                                  -1,  # [5, 8, 8, -1]
                                                  num_heads,  # 8
                                                  dropout,  # 0.1
                                                  attention_dropout,  # 0.1
                                                  use_position=False)  # DWAMFormerBlock

        self.frame_merge = MergeBlock(self.input_dim,
                                      5,  # [5, 5, 4, -1]
                                      expand=1
                                      )

        self.phoneme_merge = MergeBlock(self.input_dim,
                                        8,  # [5, 5, 4, -1]
                                        expand=1
                                        )

        self.layer_norm = nn.LayerNorm(self.input_dim)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # Define an adaptive average pooling layer that pools the input data to a length of 1.

        classifier_dim = self.input_dim  # classifier_dim = 512 * 2 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim // 2),  # 1024 -> 512
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 2, classifier_dim // 4),  # 512 -> 256
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 4, num_classes),  # 256 -> 2
        )  # Define an MLP classifier for binary classification.

    def window_size_divider(self, phonemes_info, masks):
        batch_phonemes_window_size = []
        batch_word_window_size = []

        for i in range(len(masks)):

            try:
                word_length_in_frame = []
                start_time = 0
                curr_pos = 0
                total_time = phonemes_info[i][0][-1]
                phonemes_window_size = []
                word_window_size = []
                # phoneme_s = self.phoneme_length_mlp(phonemes_info[i][-2]).squeeze(-1)
                phonemes_num = phonemes_info[i][-1]

                for word_len in range(len(phonemes_info[i][0]) - 1):
                    end_time = phonemes_info[i][0][word_len]
                    duration = end_time - start_time
                    frame_len = int(masks[i] / total_time * duration)

                    word_length_in_frame.append([curr_pos, frame_len + curr_pos])
                    start_time = end_time
                    curr_pos += frame_len + 1

                word_length_in_frame.append([curr_pos, masks[i]])

                curr_pos = 0
                curr_word = 0
                for r in range(len(word_length_in_frame)):
                    if phonemes_num[r] == 1:
                        phonemes_window_size.append(word_length_in_frame[r])
                        curr_pos = word_length_in_frame[r][1] + 1
                    else:
                        p_len = (word_length_in_frame[r][1] - word_length_in_frame[r][0]) // phonemes_num[r]

                        for p_num in range(phonemes_num[r] - 1):
                            phonemes_window_size.append([curr_pos, curr_pos + p_len])
                            curr_pos += p_len + 1

                        phonemes_window_size.append([curr_pos, word_length_in_frame[r][1]])
                        curr_pos = word_length_in_frame[r][1] + 1

                    word_window_size.append([curr_word, curr_word + phonemes_num[r]])
                    curr_word += phonemes_num[r]

            except:
                pass

            batch_phonemes_window_size.append(phonemes_window_size)
            batch_word_window_size.append(word_window_size)

        return batch_phonemes_window_size, batch_word_window_size

    def custom_avg_pooling(self, input_features, i, useage=False):
        if useage:
            n, c = input_features.shape

            if n % i != 0:
                padding_len = i - n % i
                print("padding length ", padding_len)

                input_features = F.pad(input_features, (0, 0, 0, padding_len))
                print(input_features)

                n, c = input_features.shape

            # Calculate the size of each pooling region.
            region_size = n // i
            # Calculate the length after pooling.
            pooled_length = n // region_size
            # Reshape the feature tensor to (pooled_length, region_size, c)
            reshaped_features = input_features[:pooled_length * region_size].view(pooled_length, region_size, c)
            # Compute the average along the second dimension.
            pooled_features = reshaped_features.mean(dim=1)
            return pooled_features

        else:
            return input_features.transpose(0, 1)[:i].transpose(0, 1)

    def forward(self, x, phonemes_info, masks):
        batch_phonemes_window_size, batch_word_window_size = self.window_size_divider(phonemes_info, masks)

        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        new_x = self.frame_encoder_block(x, window=batch_phonemes_window_size)
        new_x = self.frame_merge(new_x)

        new_x = self.custom_avg_pooling(x, new_x.shape[1]) + new_x
        new_x = self.layer_norm(new_x)

        new_x = self.phoneme_encoder_block(new_x, window=batch_word_window_size)
        new_x = self.phoneme_merge(new_x)

        new_x = self.custom_avg_pooling(x, new_x.shape[1]) + new_x
        new_x = self.layer_norm(new_x)

        new_x = self.word_encoder_block(new_x, window=batch_word_window_size).squeeze(dim=1)  # Remove the second dimension from x

        del batch_phonemes_window_size
        del batch_word_window_size

        new_x = self.avgpool(new_x.transpose(-1, -2)).squeeze(dim=-1)  # Transpose the last two dimensions of, then apply average pooling, and remove the final dimension.
        pred = self.classifier(new_x)  # Input x into the classifier to obtain the prediction results.

        return pred
