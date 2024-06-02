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
        self.local_size = int(local_size)  # 窗口大小 5
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
        在Transformer架构的自注意力机制中，查询（Q）、键（K）和值（V）是输入序列的不同表示，
        自注意力通过计算查询和所有键之间的点积来确定每个位置应该给予序列中其他位置多少注意力。
        点积可能会产生很大的数值，特别是当输入序列较长或特征维度较高时。
        这样的大数值在经过softmax函数时可能会导致梯度非常小，这会使训练过程变得困难，因为这可能会导致梯度消失问题。

        为了缓解这个问题，点积结果通常会除以一个缩放因子，这个缩放因子是维度的平方根的倒数（即 float(self.head_dim) ** -0.5)
        其中 self.head_dim 是键的维度大小。这种缩放有助于控制点积之后的数值范围，使得梯度在经过softmax函数时保持较为稳定。

        所以，self.scaling 的作用就是在自注意力计算中作为一个常数因子，用来缩放点积的结果，以便有助于稳定训练过程。
        """

    def get_overlap_segments(self, x: torch.Tensor, window_size: int):
        '''Get overlap segments for local attention.

        Args: 
            x: Input sequence in shape (B, T, C). 
            window_size: The needed length of the segment. Must be an odd number. 5

        本函数，将全局特征，按照窗口大小，拆分成窗口特征
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
        bsz, tgt_len = x.shape[:2]  # bsz -> batch size; tgt_len -> 特征长度

        if self.local_size == -1:  # 如果到最后的Stage
            local_size = tgt_len  # 窗口长度为特征长度
            global_attn = True
        else:  # 如果为前2个Stage
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
            # 不会允许
            need_pad = tgt_len % local_size
            if need_pad:
                pad = local_size - need_pad
                x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
                tgt_len += pad
        else:
            need_pad = 0

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)  # 对输入x，映射为不同的QKV
        Q = Q * self.scaling  # 乘以缩放因子 0.125
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        """
        Q -> (B, T, C)
        transpose(0, 1) -> (T, B, C)

        contiguous(): transpose 操作会导致张量的内存布局变得不连续，
        因此 contiguous 调用用于确保张量在内存中是连续的，这是后续 view 操作的前提。

        view(tgt_len, bsz * self.num_heads, self.head_dim) -> (T, B * 8, C // 8)

        transpose(0, 1) -> (B * 8, T, C // 8)

        把Q中的Batch，拆分成8组
        K与V一样的操作
        """
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (self.overlap) and (not global_attn):  # 前2个Stage
            Q = Q.unsqueeze(dim=2)  # (B * 8, T, C // 8) -> (B * 8, T, 1, C // 8)
            K = self.get_overlap_segments(K, window_size=local_size).transpose(-1, -2)
            """
            K -> (B * 8, T, 1, C // 8)
            get_overlap_segments -> (B * 8, T, window_size, C // 8)
            transpose(-1, -2) -> (B * 8, T, C // 8, window_size)
            """
            V = self.get_overlap_segments(V, window_size=local_size)  # (B * 8, T, window_size, C // 8)

            attn_output_weights = torch.matmul(Q, K)  # 点乘QK，获得注意力得分 (B * 8, T, 1, window_size)

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            # 每个查询对应的窗口内的键的注意力得分会被归一化，获得注意力概率 (B * 8, T, 1, window_size)

            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_output_weights, V).squeeze(dim=2)
            """
            torch.matmul(attn_output_weights, V) -> (B * 8, T, 1, C // 8)
            squeeze(dim=2) -> (B * 8, T, C // 8)
            """
        else:  # W-Stage 全局注意力
            Q = Q.contiguous().view(-1, local_size, self.head_dim)  # local_size = tgt_len
            """
            Q -> (B * 8, T, C // 8)
            view(-1, local_size, self.head_dim) -> (B * 8, T, C // 8)
            """
            K = K.contiguous().view(-1, local_size, self.head_dim)
            V = V.contiguous().view(-1, local_size, self.head_dim)

            src_len = K.size(1)  # 记录特征长度
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))  # 点乘QK，获得注意力得分 (B * 8, T, T)

            assert list(attn_output_weights.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size,
                                                        src_len]

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            # 每个查询对应的键的注意力得分会被归一化，获得注意力概率 (B * 8, T, T)

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

        将多头注意力机制的输出调整为后续层次所需的形状
        """

        attn_output = self.project_out(attn_output)  # 全连接转换

        if need_pad:
            attn_output = attn_output[:, :-pad, :]

        return attn_output


class DWAMFormerEncoder(nn.Module):
    def __init__(self,
                 embed_dim,  # 输入特征的维度 512
                 ffn_embed_dim=2304,  # ffn维度 256
                 local_size=0,  # 窗口大小 5
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
        self.dropout = dropout  # 定义dropout率  0.1
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
        x = self.add_position(x, x_position)  # 在输入x上，加入位置编码

        x = self.attention(x, window)  # 将x输入到MSA中
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout层（只在训练阶段启用）
        x = residual + x  # 残差
        x = self.attention_layer_norm(x)  # LayerNom 层规则化

        residual = x

        x = self.activation_fn(self.fc1(x))  # 全连接，后过激活 512 -> 256
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout层（只在训练阶段启用）
        x = self.fc2(x)  # 全连接 256 -> 512
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout层（只在训练阶段启用）
        x = residual + x  # 残差
        x = self.final_layer_norm(x)  # LayerNom 层规则化
        return x
