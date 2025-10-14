import torch
from torch import nn
import torch.nn.functional as F

from src.model.pblstm_encoder import PBLSTMEncoder
from src.model.rnn_decoder import RNNDecoder


class LasModel(nn.Module):
    """
    Listen, Attend and Spell model for ASR
    """

    def __init__(self, n_feats, n_tokens, enc_hidden=256, dec_hidden=512, 
                 enc_layers=3, dec_layers=2, enc_dropout=0.2):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            enc_hidden (int): number of hidden features in the encoder.
            dec_hidden (int): number of hidden features in the decoder.
            enc_layers (int): number of layers in the encoder.
            dec_layers (int): number of layers in the decoder.
            enc_dropout (float): dropout probability in the encoder.
            dec_dropout (float): dropout probability in the decoder.
        """
        super().__init__()
        
        self.encoder = PBLSTMEncoder(n_feats, enc_hidden, enc_layers, enc_dropout)
        self.decoder = RNNDecoder(n_tokens, enc_hidden * 2, dec_hidden, dec_layers)
        
        self.n_tokens = n_tokens
        self.time_reduction_factor = 2 ** (enc_layers - 1)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        encoder_outputs, encoder_lengths = self.encoder(spectrogram, spectrogram_length)
        
        log_probs = self.decoder(encoder_outputs, encoder_lengths)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate output lengths after time reduction in the encoder.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return torch.ceil(input_lengths.float() / self.time_reduction_factor).int()

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
    

class Attention(nn.Module):
    """
    Attention mechanism for the LAS model.
    """
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        """
        Calculate attention weights and context vector.
        
        Args:
            encoder_outputs: Outputs from encoder [batch, enc_len, enc_dim]
            decoder_hidden: Current decoder hidden state [batch, dec_dim]
            mask: Mask for padding positions
            
        Returns:
            context: Context vector [batch, enc_dim]
            attention_weights: Attention weights [batch, enc_len]
        """
        _, enc_len, _ = encoder_outputs.size()
        
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, enc_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Speller(nn.Module):
    """
    LSTM-based decoder with attention for the LAS model.
    """
    def __init__(self, vocab_size, enc_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + enc_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attention = Attention(enc_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim + enc_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_outputs, encoder_lengths):
        """
        Simplified forward pass that produces log probabilities for the whole sequence.
        For real deployment, you'd use an autoregressive approach.
        
        Args:
            encoder_outputs: Outputs from encoder [batch, enc_len, enc_dim]
            encoder_lengths: Lengths of encoder outputs
            
        Returns:
            log_probs: Log probabilities [batch, max_len, vocab_size]
        """
        batch_size = encoder_outputs.size(0)
        max_enc_len = encoder_outputs.size(1)
        
        max_len = max_enc_len
        
        hidden = self._init_hidden(batch_size, encoder_outputs.device)
        
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_outputs.device)
        
        outputs = []
        
        for _ in range(max_len):
            embedded = self.embedding(decoder_input).squeeze(1)
            
            hidden_for_attn = hidden[0][-1]
            
            context, _ = self.attention(encoder_outputs, hidden_for_attn)
            
            lstm_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
            
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            
            output = torch.cat((lstm_output.squeeze(1), context), dim=1)
            prediction = self.fc_out(output)
            
            outputs.append(prediction)
            
            decoder_input = prediction.argmax(1).unsqueeze(1)
        
        logits = torch.stack(outputs, dim=1)
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs
    
    def _init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        )
