import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCConformerModel(nn.Module):
    """
    CTC-based Conformer model for ASR, combining CNNs and Transformers
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        enc_dim=256,
        dec_dim=640,
        enc_layers=4,
        dec_layers=1,
        enc_heads=4,
        enc_ff_mult=4,
        enc_kernel_size=31,
        dropout=0.1,
        subsampling_factor=4,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            enc_dim (int): dimension of the encoder layers.
            dec_dim (int): dimension of the decoder layers.
            enc_layers (int): number of Conformer blocks.
            dec_layers (int): number of decoder layers.
            enc_heads (int): number of attention heads.
            enc_ff_dim (int): dimension of feed-forward network.
            enc_kernel_size (int): kernel size for depthwise convolution.
            dropout (float): dropout probability.
            subsampling_factor (int): factor to reduce the time dimension.
        """
        super().__init__()
        
        # Input subsampling (reduces time dimension)
        self.subsampling = SubsamplingBlock(n_feats, enc_dim, subsampling_factor)
        self.subsampling_factor = subsampling_factor
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=enc_dim,
                ff_dim=enc_ff_mult * enc_dim,
                n_heads=enc_heads,
                kernel_size=enc_kernel_size,
                dropout=dropout
            ) for _ in range(enc_layers)
        ])
        
        self.lstm = nn.LSTM(enc_dim, dec_dim, batch_first=True, num_layers=dec_layers, bidirectional=True)
        self.ctc_projection = nn.Linear(enc_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram [batch, time, features].
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.mean(1).transpose(1, 2)  # Ensure [B, T, F]
        max_len = spectrogram.size(1)
        mask = torch.arange(max_len, device=spectrogram.device).expand(len(spectrogram_length), max_len) < spectrogram_length.unsqueeze(1).to(spectrogram.device)
        
        x, mask = self.subsampling(spectrogram, mask)
        
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        logits = self.ctc_projection(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get transformed lengths
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate output lengths after subsampling.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return torch.div(input_lengths, self.subsampling_factor, rounding_mode='floor')

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class SubsamplingBlock(nn.Module):
    """
    Subsampling block to reduce the sequence length
    """
    def __init__(self, input_dim, output_dim, subsampling_factor=4):
        super().__init__()
        
        self.subsampling_factor = subsampling_factor
        
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        
        if subsampling_factor >= 4:
            self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
            self.relu2 = nn.ReLU()
            
        if subsampling_factor == 8:
            self.conv3 = nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
            self.relu3 = nn.ReLU()
        
        self.proj = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
            mask (Tensor): Padding mask [batch, time]
        Returns:
            x (Tensor): Subsampled tensor [batch, time//factor, output_dim]
            mask (Tensor): Updated mask [batch, time//factor]
        """
        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu1(x)
        mask = mask[:, ::2]
        
        if self.subsampling_factor >= 4:
            x = self.conv2(x)
            x = self.relu2(x)
            mask = mask[:, ::2]
            
        if self.subsampling_factor == 8:
            x = self.conv3(x)
            x = self.relu3(x)
            mask = mask[:, ::2]
        
        # [B, F, T] -> [B, T, F]
        x = x.transpose(1, 2)
        
        x = self.proj(x)
        x = self.layer_norm(x)
        
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
        
        self.layer_norm = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features]
            mask (Tensor): Attention mask [batch, time]
        Returns:
            output (Tensor): Output tensor [batch, time, features]
        """
        x = self.layer_norm(x)
        
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        
        output, _ = self.mhsa(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
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
        
        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # [B, F, T] -> [B, T, F]
        x = x.transpose(1, 2)
        
        return x


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
