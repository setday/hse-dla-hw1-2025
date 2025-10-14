import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder for ASR
    """

    def __init__(
        self,
        n_feats,
        dim=256,
        n_layers=4,
        n_heads=4,
        ff_mult=4,
        kernel_size=31,
        dropout=0.1,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            dim (int): dimension of the encoder layers.
            n_layers (int): number of Conformer blocks.
            n_heads (int): number of attention heads.
            ff_mult (int): dimension of feed-forward network.
            kernel_size (int): kernel size for depthwise convolution.
            dropout (float): dropout probability.
            subsampling_factor (int): factor to reduce the time dimension.
        """
        super().__init__()
        
        # Input subsampling (reduces time dimension)
        self.subsampling = SubsamplingBlock(1, dim)
        self.linear_proj = nn.Linear(dim * (((n_feats - 1) // 2 - 1) // 2), dim)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                ff_dim=ff_mult * dim,
                n_heads=n_heads,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def forward(self, spectrogram, mask=None):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram [batch, time, features].
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            x (Tensor): output tensor [batch, time//factor, dim].
        """
        x, mask = self.subsampling(spectrogram, mask)
        x = self.linear_proj(x)

        if mask is not None:
            mask = torch.min(mask[:, None, :], mask[:, :, None])

        for block in self.conformer_blocks:
            x = block(x, mask)
        
        return x


class SubsamplingBlock(nn.Module):
    """
    Subsampling block to reduce the sequence length
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor [batch, modalities, time, features]
            mask (Tensor): Padding mask [batch, time]
        Returns:
            x (Tensor): Subsampled tensor [batch, time//factor, output_dim]
            mask (Tensor): Updated mask [batch, time//factor]
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        mask = mask[:, :-2:2]
        mask = mask[:, :-2:2]

        x = x.transpose(1, 2).flatten(2)

        return x, mask


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN-MHSA-Conv-FFN with residual connections
    """
    def __init__(self, dim, ff_dim, n_heads, kernel_size, dropout=0.1):
        super().__init__()
        
        self.ff1 = FeedForwardModule(dim, ff_dim, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(dim, n_heads, dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ff2 = FeedForwardModule(dim, ff_dim, dropout)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
            mask (Tensor): Attention mask [batch, time]
        Returns:
            x (Tensor): Output tensor [batch, time, features]
        """
        x = x + self.ff1(x) * 0.5
        x = x + self.mhsa(x, mask)
        x = x + self.conv(x)
        x = x + self.ff2(x) * 0.5
        x = self.layer_norm(x)
        
        return x


class FeedForwardModule(nn.Module):
    """
    Feed-forward module in Conformer block
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
        Returns:
            x (Tensor): Output tensor [batch, time, features]
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-head self-attention module in Conformer block
    """
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.d_head = dim // n_heads
        self.num_heads = n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_pos = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim)

        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)
        
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        encodings = torch.zeros(10000, dim)
        pos = torch.arange(0, 10000, dtype=torch.float)
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.positional_encodings = nn.Parameter(encodings, requires_grad=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
            mask (Tensor): Attention mask [batch, time]
        Returns:
            output (Tensor): Output tensor [batch, time, features]
        """
        batch_size, seq_length, _ = x.size()

        x = self.layer_norm(x)
        
        q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
        k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1)
        v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1)
        
        pos_emb = self.positional_encodings[:seq_length, :].repeat(batch_size, 1, 1)
        pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1)

        QK = torch.matmul((q + self.u).transpose(1, 2), k)
        QP = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        
        batch_size, num_heads, seq_length1, seq_length2 = QP.size()
        zeros = QP.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_emb = torch.cat([zeros, QP], dim=-1)
        padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        BD = padded_emb[:, :, 1:].view_as(QP)

        attn = (QK + BD) / math.sqrt(self.dim)

        if mask is not None:
            mask_value = -1e30 if attn.dtype == torch.float32 else -1e4
            attn.masked_fill_(~mask.unsqueeze(1), mask_value)

        attn = F.softmax(attn, -1)

        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.dim)
        output = self.W_out(output)

        output = self.dropout(output)
        
        return output


class ConvolutionModule(nn.Module):
    """
    Convolution module in Conformer block
    """
    def __init__(self, dim, kernel_size, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=padding, groups=dim
        )
        
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
        Returns:
            x (Tensor): Output tensor [batch, time, features]
        """
        x = self.layer_norm(x)
        
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        
        return x


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
