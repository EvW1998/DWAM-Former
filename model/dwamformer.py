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

        out_channels = in_channels * expand  # 维度扩展 512
        self.MS = int(merge_scale)  # 5
        # self.pool = nn.AdaptiveAvgPool2d((1, in_channels))  # 2维平均池化层，窗口大小 (1, 512)
        self.attn_pool = AttentionPool(in_channels * self.MS)
        self.fc = nn.Linear(in_channels, out_channels)  # 全连接扩展维度
        self.norm = nn.LayerNorm(out_channels)  # 层规范化

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS  # ms = 5

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        """
        确保输入特征的长度，能被ms整除，如无法被整除，则往输入特征后，填充0，以使特征能被ms整除
        """

        x = x.view(B, T // ms, ms, C)  # (B, T, C) -> (B, T // ms, ms, C)
        # x = self.pool(x).squeeze(dim=-2)
        x = self.attn_pool(x)
        """
        self.pool(x) -> (B, T // ms, 1, C)
        squeeze(dim=-2) -> (B, T // ms, C)

        将ms个特征，平均池化为1个特征
        
        """
        x = self.norm(self.fc(x))
        """
        self.fc(x) -> (B, T // ms, C * expand) 特征升维
        """

        return x


class DWAMFormerBlock(nn.Module):
    def __init__(self,
                 num_layers,  # 层数 2
                 embed_dim,  # 输入特征维度 512
                 ffn_embed_dim=2304,  # ffn 维度 256
                 local_size=0,  # 窗口大小 Tw 5
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
        self.position = create_PositionalEncoding(embed_dim) if use_position else None  # 生成位置编码 (2000, 512)
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
                Xavier初始化是为了在训练开始时保持每层输入和输出的方差一致，以避免梯度在深度网络中消失或爆炸

                这种初始化方法设置参数的值，使得每个参数的值都均匀分布在一个范围内，这个范围是基于参数张量维度计算出的。
                它有助于保持激活函数的方差在前向传播和反向传播时保持不变。
                """

    def forward(self, x, window):
        output = self.input_norm(x)  # 对输入x进行LayerNorm

        for layer in self.layers:
            output = layer(output, window, self.position)  # 将规则化后的输入，以及位置编码，输入到Encoder中

        return output


class DWAMFormer(nn.Module):
    """
    DWAM-Former is a transformer structure. By three different stages, as frame stage, phoneme stage, and word stage.
    """

    def __init__(self,
                 input_dim,  # 输入特征的维度（转换成能被num_heads，8，整除的维度）
                 ffn_embed_dim,  # MSA中ffn的embedding维度（input_dim//2）
                 num_layers,  # F-Stage, P-Stage, W-Stage和U-Stage中，Block的层数 [N1, N2, N3, N4]
                 num_heads,  # MSA中的注意力头数
                 hop,  # hop length
                 num_classes,  # 分类数
                 expand,  # Merging block中维度拓展 [1, 1, 1, -1] 或 [1, 1, 2, -1]
                 dropout=0.1, attention_dropout=0.1, device='cuda', _ms=None, **kwargs):

        """
        模型初始化

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

        assert isinstance(num_layers, list)  # 确认num_layers为一个list

        self.frame_encoder_block = DWAMFormerBlock(3,  # [2, 2, 4, 4]
                                                   self.input_dim,  # [512, 512, 512, 1024]
                                                   ffn_embed_dim,  # [256, 256, 256, 512]
                                                   5,  # [5, 8, 8, -1]
                                                   num_heads,  # 8
                                                   dropout,  # 0.1
                                                   attention_dropout,  # 0.1
                                                   use_position=True)  # 定义了一个DWAMFormerBlock

        self.phoneme_encoder_block = DWAMFormerBlock(3,  # [2, 2, 4, 4]
                                                     self.input_dim,  # [512, 512, 512, 1024]
                                                     ffn_embed_dim,  # [256, 256, 256, 512]
                                                     8,  # [5, 8, 8, -1]
                                                     num_heads,  # 8
                                                     dropout,  # 0.1
                                                     attention_dropout,  # 0.1
                                                     use_position=False)  # 定义了一个DWAMFormerBlock

        self.word_encoder_block = DWAMFormerBlock(6,  # [2, 2, 4, 4]
                                                  self.input_dim,  # [512, 512, 512, 1024]
                                                  ffn_embed_dim,  # [256, 256, 256, 512]
                                                  -1,  # [5, 8, 8, -1]
                                                  num_heads,  # 8
                                                  dropout,  # 0.1
                                                  attention_dropout,  # 0.1
                                                  use_position=False)  # 定义了一个DWAMFormerBlock

        self.frame_merge = MergeBlock(self.input_dim,
                                      5,  # [5, 5, 4, -1]
                                      expand=1
                                      )

        self.phoneme_merge = MergeBlock(self.input_dim,
                                        8,  # [5, 5, 4, -1]
                                        expand=1
                                        )

        self.layer_norm = nn.LayerNorm(self.input_dim)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 定义一个自适应平均池化层，将输入数据，池化为长度为1的数据

        classifier_dim = self.input_dim  # classifier_dim = 512 * 2 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim // 2),  # 1024 -> 512
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 2, classifier_dim // 4),  # 512 -> 256
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 4, num_classes),  # 256 -> 2
        )  # 定义了一个MLP分类器（二分类）

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

            # 计算每个池化区域的大小
            region_size = n // i
            # 计算池化后的长度
            pooled_length = n // region_size
            # 将特征张量重塑为 (pooled_length, region_size, c)
            reshaped_features = input_features[:pooled_length * region_size].view(pooled_length, region_size, c)
            # 沿着第二个维度求平均值
            pooled_features = reshaped_features.mean(dim=1)
            return pooled_features

        else:
            return input_features.transpose(0, 1)[:i].transpose(0, 1)

    def forward(self, x, phonemes_info, masks):
        batch_phonemes_window_size, batch_word_window_size = self.window_size_divider(phonemes_info, masks)

        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]  # x (batch, ？？, 特征维度)

        new_x = self.frame_encoder_block(x, window=batch_phonemes_window_size)
        new_x = self.frame_merge(new_x)

        new_x = self.custom_avg_pooling(x, new_x.shape[1]) + new_x
        new_x = self.layer_norm(new_x)

        new_x = self.phoneme_encoder_block(new_x, window=batch_word_window_size)
        new_x = self.phoneme_merge(new_x)

        new_x = self.custom_avg_pooling(x, new_x.shape[1]) + new_x
        new_x = self.layer_norm(new_x)

        new_x = self.word_encoder_block(new_x, window=batch_word_window_size).squeeze(dim=1)  # 将x去掉第2个维度

        del batch_phonemes_window_size
        del batch_word_window_size

        new_x = self.avgpool(new_x.transpose(-1, -2)).squeeze(dim=-1)  # 将x的最后两个维度做转置，avgpool后，去掉最后一个维度
        pred = self.classifier(new_x)  # 将x输入分类器中，得到预测结果

        return pred
