"""
@author: Xiaoping Yue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.utils import _get_activation_fn


class Speech_MSA(nn.Module):
    ''' Speech-based Multi-Head Self-Attention (Speech-MSA)
    
    Input dimension order is (batch_size, seq_len, input_dim).
    '''

    def __init__(self,
                 embed_dim,  # 512
                 num_heads,  # 8
                 local_size,  # 5
                 dropout=0.,  # 0.1
                 bias=True,
                 overlap=False  # True
                 ):
        """
        embed_dim, # 512
        num_heads,  # 8
        local_size, # 5
        attention_dropout, # 0.1
        overlap=overlap # True
        """

        super(Speech_MSA, self).__init__()
        self.qdim = embed_dim  # 512
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.local_size = int(local_size)  # Window size 5
        self.overlap = overlap  # overlap = True may have nondeterministic behavior.

        self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)  # 512 -> 1536
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)  # 512 -> 512

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads  # 64
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5  # 缩放因子 0.125
        """
        In the self-attention mechanism of the Transformer architecture, queries (Q), keys (K), and values (V) 
        are different representations of the input sequence. Self-attention determines how much attention each position 
        should give to other positions in the sequence by calculating the dot product between the query and all keys. 
        The dot product can yield very large values, especially when the input sequence is long or the feature dimension is high. 
        These large values can result in very small gradients when passed through the softmax function, 
        making the training process difficult due to the potential for gradient vanishing.

        To alleviate this issue, the dot product result is typically divided by a scaling factor, 
        which is the inverse of the square root of the dimension (i.e., float(self.head_dim) ** -0.5), 
        where self.head_dim is the dimensionality of the keys. This scaling helps control the range of values after the dot product, 
        ensuring that the gradients remain relatively stable when passed through the softmax function.

        Therefore, the purpose of self.scaling in self-attention computation is to act as a constant factor to scale the dot product results, 
        thereby stabilizing the training process.
        """

    def get_overlap_segments(self, x: torch.Tensor, window_size: int):
        '''Get overlap segments for local attention.

        Args: 
            x: Input sequence in shape (B, T, C). 
            window_size: The needed length of the segment. Must be an odd number. 5

        This function splits the global features into window features according to the window size.
        '''
        # assert window_size % 2, f'window_size must be an odd number, but get {window_size}.'
        if not window_size % 2:
            window_size += 1  # window_size must be an odd number

        b, t, c = x.shape
        pad_len = (window_size - 1) // 2  # T_w / 2
        x = F.pad(x, (0, 0, pad_len, pad_len), value=0)  # (B, T, C) -> (B, T + 2 * pad_len, C)

        stride = x.stride()
        out_shape = (b, t, window_size, c)
        out_stride = (stride[0], stride[1], stride[1], stride[2])

        return torch.as_strided(x, size=out_shape, stride=out_stride)  # (B, T, window_size, C)

    def forward(self, x, window):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len = x.shape[:2]  # bsz -> batch size; tgt_len -> feature length

        if self.local_size == -1:  # if in the last Stage
            local_size = tgt_len  # The window length is equal to the feature length
            global_attn = True
        else:  # if in first 2 Stage
            local_size = self.local_size
            # largest_window_size = self.local_size
            # for ws in window:
            #     for w in ws:
            #         w_size = w[1] - w[0]
            #         if w_size > largest_window_size:
            #             largest_window_size = w_size
            #
            # local_size = largest_window_size

            global_attn = False

        if not self.overlap:
            need_pad = tgt_len % local_size
            if need_pad:
                pad = local_size - need_pad
                x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
                tgt_len += pad
        else:
            need_pad = 0

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1) 
        Q = Q * self.scaling  
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        """
        Q -> (B, T, C)
        transpose(0, 1) -> (T, B, C)

        The `contiguous()` function is called because the `transpose` operation can make the tensor's memory layout non-contiguous. 
        Therefore, `contiguous()` ensures that the tensor is stored in a contiguous block of memory, 
        which is a prerequisite for subsequent `view` operations.

        view(tgt_len, bsz * self.num_heads, self.head_dim) -> (T, B * 8, C // 8)

        transpose(0, 1) -> (B * 8, T, C // 8)
        """
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (self.overlap) and (not global_attn):
            Q = Q.unsqueeze(dim=2)  # (B * 8, T, C // 8) -> (B * 8, T, 1, C // 8)
            K = self.get_overlap_segments(K, window_size=local_size).transpose(-1, -2)
            """
            K -> (B * 8, T, 1, C // 8)
            get_overlap_segments -> (B * 8, T, window_size, C // 8)
            transpose(-1, -2) -> (B * 8, T, C // 8, window_size)
            """
            V = self.get_overlap_segments(V, window_size=local_size)  # (B * 8, T, window_size, C // 8)

            attn_output_weights = torch.matmul(Q, K)  # Attention weight (B * 8, T, 1, window_size)

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            # The attention scores of the keys within the window corresponding to each query will be normalized to obtain the attention probabilities.
            # (B * 8, T, 1, window_size)

            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_output_weights, V).squeeze(dim=2)
            """
            torch.matmul(attn_output_weights, V) -> (B * 8, T, 1, C // 8)
            squeeze(dim=2) -> (B * 8, T, C // 8)
            """
        else:  # W-Stage Global attention
            Q = Q.contiguous().view(-1, local_size, self.head_dim)  # local_size = tgt_len
            """
            Q -> (B * 8, T, C // 8)
            view(-1, local_size, self.head_dim) -> (B * 8, T, C // 8)
            """
            K = K.contiguous().view(-1, local_size, self.head_dim)
            V = V.contiguous().view(-1, local_size, self.head_dim)

            src_len = K.size(1) 
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))

            assert list(attn_output_weights.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size,
                                                        src_len]

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, V)  # (B * 8, T, C // 8)

            assert list(attn_output.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size, self.head_dim]
            attn_output = attn_output.view(bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        """
        atten_output -> (B * 8, T, C // 8)
        transpose(0, 1) -> (T, B * 8, C // 8)
        view(tgt_len, bsz, self.embed_dim) -> view(T, B, C) -> (T, B, C)
        transpose(0, 1) -> (B, T, C)

        Adjust the output of the multi-head attention mechanism to the shape required by the subsequent layers.
        """

        attn_output = self.project_out(attn_output)

        if need_pad:
            attn_output = attn_output[:, :-pad, :]

        return attn_output


class DWAMFormerEncoder(nn.Module):
    def __init__(self,
                 embed_dim,  # 512
                 ffn_embed_dim=2304,  # 256
                 local_size=0,  # 5
                 num_heads=8,  # 8
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation='relu',
                 overlap=False):  # True
        """
        embed_dim, # 512
        ffn_embed_dim, # 256
        local_size, # 5
        num_heads, # 8
        dropout, # 0.1
        attention_dropout, # 0.1
        activation, # 'relu'
        overlap=True
        """

        super().__init__()
        self.dropout = dropout  #  0.1
        self.activation_fn = _get_activation_fn(activation)  # F.relu

        self.attention = Speech_MSA(embed_dim,  # 512
                                    num_heads,  # 8
                                    local_size,  # 5
                                    attention_dropout,  # 0.1
                                    overlap=overlap  # True
                                    )

        self.attention_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)  # 512 -> 256
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)  # 256 -> 512

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def add_position(self, x, position=None, mask=None):
        '''add position information to the input x

        x: B, T, C
        position: 2000, C
        mask: B, T
        '''
        if position is None:
            return x
        else:
            B, T = x.shape[:2]
            position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
            position = position * ((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
            return x + position

    def forward(self, x, window, x_position=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.

        x: B, T, C
        """
        residual = x
        x = self.add_position(x, x_position)  # Add positional encoding to the input x.

        x = self.attention(x, window)  # Input x into the Multi-Head Self-Attention (MSA) mechanism.
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout layer (enabled only during the training phase)
        x = residual + x  # residual
        x = self.attention_layer_norm(x)  # LayerNom

        residual = x

        x = self.activation_fn(self.fc1(x))  # Full connection 512 -> 256
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout layer (enabled only during the training phase)
        x = self.fc2(x)  # Full connection 256 -> 512
        x = F.dropout(x, p=self.dropout, training=self.training)  # dDropout layer (enabled only during the training phase)
        x = residual + x  # residual
        x = self.final_layer_norm(x)  # LayerNom
        return x
