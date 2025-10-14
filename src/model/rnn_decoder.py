import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    """
    RNN-based Decoder for ASR
    """

    def __init__(
        self,
        n_tokens,
        in_dim=256,
        out_dim=640,
        n_layers=1,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            in_dim (int): dimension of the encoder layers.
            out_dim (int): dimension of the decoder layers.
            n_layers (int): number of decoder layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True, num_layers=n_layers)
        self.ctc_projection = nn.Linear(out_dim, n_tokens)

    def forward(self, x, x_length):
        """
        Model forward method.

        Args:
            hidden_states (Tensor): input encoder features [batch, time, features].
            hidden_states_length (Tensor): lengths of the input sequences [batch].
        Returns:
            log_probs (Tensor): log probabilities over tokens [batch, time, n_tokens].
            out_length (Tensor): output lengths [batch].
        """
        x = nn.utils.rnn.pack_padded_sequence(x, x_length.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, out_length = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            
        logits = self.ctc_projection(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, out_length
