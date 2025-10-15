from typing import Optional
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
        self.decoder = Speller(n_tokens, enc_hidden, dec_hidden, dec_layers)
        
        self.n_tokens = n_tokens
        self.time_reduction_factor = 2 ** (enc_layers - 1)

    def forward(self, spectrogram, spectrogram_length, text_encoded, text_encoded_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.mean(dim=1).transpose(1, 2)  # [B, T, F]
        encoder_outputs, _ = self.encoder(spectrogram, spectrogram_length)
        
        if text_encoded is None:
            log_probs = self.decoder.generate(encoder_outputs)
        else:
            log_probs = self.decoder(encoder_outputs, text_encoded)

        if text_encoded_length is None:
            text_encoded_length = torch.full(
                (spectrogram.size(0),),
                log_probs.size(1),
                dtype=torch.int32,
                device=log_probs.device
            )
        
        return {"log_probs": log_probs, "log_probs_length": text_encoded_length}

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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Calculate attention weights and context vector.
        
        Args:
            encoder_outputs: Outputs from encoder [batch, enc_len, enc_dim]
            decoder_hidden: Current decoder hidden state [batch, dec_dim]
            mask: Mask for padding positions
            
        Returns:
            context: Context vector [batch, enc_dim]
        """
        energy = torch.bmm(
            self.phi(encoder_outputs),
            self.psi(decoder_hidden).unsqueeze(2)
        ).squeeze(2)
        attention_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context


class Speller(nn.Module):
    """
    LSTM-based decoder with attention for the LAS model.
    """
    def __init__(self, vocab_size, enc_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size
        
        self.lstm = nn.LSTM(vocab_size + hidden_dim + 1, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attention = Attention(2 * enc_dim, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size + 1)
        )

    def step(self, encoder_outputs: torch.Tensor, prev_token_distribution: torch.Tensor,
             prev_rnn_state: Optional[torch.Tensor] = None, attention_context: Optional[torch.Tensor] = None):
        """
        Single decoding step.

        Args:
            encoder_outputs (Tensor): Outputs from encoder [batch, enc_len, enc_dim]
            prev_token_distribution (Tensor): Previous token indices [batch]
            prev_rnn_state (Tuple[Tensor, Tensor], optional): Previous RNN hidden and cell states
            attention_context (Tensor, optional): Previous attention context vector [batch, enc_dim]
        Returns:
            output (Tensor): Logits for the current time step [batch, vocab_size]
            rnn_state (Tuple[Tensor, Tensor]): Current RNN hidden and cell states
            attention_context (Tensor): Current attention context vector [batch, enc_dim]
        """
        if attention_context is None:
            attention_context = torch.zeros(encoder_outputs.size(0), self.hidden_dim, device=encoder_outputs.device)
        
        lstm_input = torch.cat((prev_token_distribution, attention_context), dim=-1)
        lstm_output, rnn_state = self.lstm(lstm_input, prev_rnn_state)
        
        attention_context = self.attention(encoder_outputs, lstm_output)
        
        output = torch.cat((lstm_output, attention_context), dim=1)
        output = self.fc_out(output)
        
        return output, rnn_state, attention_context

    def forward(self, encoder_outputs: torch.Tensor, ground_truth: torch.Tensor):
        """
        Simplified forward pass that produces log probabilities for the whole sequence.
        For real deployment, you'd use an autoregressive approach.
        
        Args:
            encoder_outputs: Outputs from encoder [batch, enc_len, enc_dim]
            encoder_lengths: Lengths of encoder outputs
            
        Returns:
            log_probs: Log probabilities [batch, max_len, vocab_size]
        """
        sos_id = self.vocab_size  # Start-of-sequence token ID

        batch_size = encoder_outputs.size(0)
        max_len = ground_truth.size(1)
        device = encoder_outputs.device

        ground_truth = torch.nn.functional.pad(ground_truth, (1, 0), value=sos_id)  # Pad with SOS
        
        inputs_encoded = torch.zeros((batch_size, max_len + 1, self.vocab_size + 1), device=device)
        inputs_encoded.scatter_(2, ground_truth.unsqueeze(2).to(torch.int32), 1)

        inputs = inputs_encoded[:, 0]  # Initial input is SOS
        rnn_state, attention_context = None, None
        
        log_probs = []
        
        for step in range(max_len):
            output, rnn_state, attention_context = self.step(
                encoder_outputs, inputs, rnn_state, attention_context
            )
            log_prob = F.log_softmax(output, dim=1)
            log_probs.append(log_prob.unsqueeze(1))

            if ground_truth is not None and torch.rand((1,)).item() < 0.9:
                inputs = inputs_encoded[:, step + 1]  # Teacher forcing
            else:
                # Sample the next input from the output distribution
                dist = torch.distributions.Categorical(logits=log_prob)
                inputs = torch.zeros((batch_size, self.vocab_size + 1), device=device)
                inputs.scatter_(1, dist.sample().unsqueeze(1), 1)
        
        log_probs = torch.cat(log_probs, dim=1)
        return log_probs
    
    def generate(self, encoder_outputs: torch.Tensor, max_len: int = 300):
        """
        Generate sequence using greedy decoding.

        Args:
            encoder_outputs (Tensor): Outputs from encoder [batch, enc_len, enc_dim]
            max_len (int): Maximum length of the generated sequence
        Returns:
            generated_ids (Tensor): Generated token IDs [batch, max_len]
        """
        sos_id = self.vocab_size  # Start-of-sequence token ID

        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        inputs = torch.zeros((batch_size, 1, self.vocab_size + 1), device=device)
        inputs.scatter_(2, torch.full((batch_size, 1), sos_id, device=device), 1)

        rnn_state, attention_context = None, None
        
        generated_ids = []

        for _ in range(max_len):
            output, rnn_state, attention_context = self.step(
                encoder_outputs, inputs, rnn_state, attention_context
            )
            log_prob = F.log_softmax(output, dim=1)
            predicted_ids = torch.argmax(log_prob, dim=1)
            generated_ids.append(predicted_ids.unsqueeze(1))

            if (predicted_ids == 0).all():
                break

            inputs = torch.zeros((batch_size, 1, self.vocab_size + 1), device=device)
            inputs.scatter_(2, predicted_ids.unsqueeze(1), 1)

        generated_ids = torch.cat(generated_ids, dim=1)

        return generated_ids
    
    def generate_beam_search(self, encoder_outputs: torch.Tensor, beam_size: int = 5, max_len: int = 300):
        """
        Generate sequence using beam search decoding.

        Args:
            encoder_outputs (Tensor): Outputs from encoder [batch, enc_len, enc_dim]
            beam_size (int): Beam size for beam search
            max_len (int): Maximum length of the generated sequence
        Returns:
            generated_ids (List[List[int]]): Generated token IDs for each batch item
        """
        sos_id = self.vocab_size  # Start-of-sequence token ID

        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        generated_ids = []

        for b in range(batch_size):
            beams = [( [sos_id], 0.0, None, None )]  # (tokens, score, rnn_state, attention_context)

            for _ in range(max_len):
                new_beams = []
                for tokens, score, rnn_state, attention_context in beams:
                    inputs = torch.zeros((1, 1, self.vocab_size + 1), device=device)
                    inputs.scatter_(2, torch.tensor([[tokens[-1]]], device=device), 1)

                    output, new_rnn_state, new_attention_context = self.step(
                        encoder_outputs[b:b+1], inputs, rnn_state, attention_context
                    )
                    log_prob = F.log_softmax(output, dim=1)
                    topk_log_probs, topk_ids = torch.topk(log_prob, beam_size)

                    for k in range(beam_size):
                        new_tokens = tokens + [topk_ids[0][k].item()]
                        new_score = score + topk_log_probs[0][k].item()
                        new_beams.append((new_tokens, new_score, new_rnn_state, new_attention_context))

                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            best_beam = max(beams, key=lambda x: x[1])
            generated_ids.append(best_beam[0][1:])  # Exclude SOS

        return generated_ids
