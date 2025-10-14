import torch
import torch.nn as nn


from src.model.conformer_encoder import ConformerEncoder
from src.model.rnn_decoder import RNNDecoder


class CTCConformerModel(nn.Module):
    """
    RNN-based Conformer model for ASR, combining CNNs and Transformers
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

        self.subsampling_factor = 4
        
        self.encoder = ConformerEncoder(
            n_feats=n_feats,
            dim=enc_dim,
            n_layers=enc_layers,
            n_heads=enc_heads,
            ff_mult=enc_ff_mult,
            kernel_size=enc_kernel_size,
            dropout=dropout,
        )
        self.decoder = RNNDecoder(
            n_tokens=n_tokens,
            in_dim=enc_dim,
            out_dim=dec_dim,
            n_layers=dec_layers,
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram [batch, modalities, features, time].
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        batch_size = spectrogram.size(0)
        max_len = spectrogram.size(-1)
        device = spectrogram.device
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < spectrogram_length.unsqueeze(1).to(device)
        
        spectrogram = spectrogram.transpose(-1, -2)
        x = self.encoder(spectrogram, mask)
        log_probs, log_probs_length = self.decoder(x, self.transform_input_lengths(spectrogram_length))
        
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate output lengths after subsampling.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return ((input_lengths // 2) - 1) // 2 - 1

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
