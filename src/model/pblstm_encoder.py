import torch
import torch.nn as nn


class PBLSTMEncoder(nn.Module):
    """
    Pyramidal BiLSTM-based Encoder for ASR
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        """
        Args:
            input_dim (int): dimension of the input features.
            hidden_dim (int): dimension of the encoder layers.
            num_layers (int): number of encoder layers.
            dropout (float): dropout rate.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        ])
        
        for _ in range(1, num_layers):
            self.lstm_layers.append(
                nn.LSTM(hidden_dim * 2 * 2, hidden_dim, 1, bidirectional=True, batch_first=True)
            )
        
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, lengths):
        """
        Args:
            x (Tensor): Spectrogram [batch, time, features]
            lengths (Tensor): Original lengths of input sequences
        Returns:
            outputs (Tensor): Encoded features
            lengths (Tensor): Updated lengths after time reduction
        """
        for i, lstm in enumerate(self.lstm_layers):
            x = self.dropout(x)
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_outputs, _ = lstm(packed_x)
            x, lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            
            if i < self.num_layers - 1:
                if x.size(1) % 2 != 0:
                    x = torch.cat([x, torch.zeros_like(x[:, -1:])], dim=1)
                
                batch_size, time_steps, features = x.size()
                x = x.contiguous().view(batch_size, time_steps // 2, features * 2)
                lengths = torch.ceil(lengths.float() / 2).int()
        
        return x, lengths
