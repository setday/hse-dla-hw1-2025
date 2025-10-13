from typing import List, Literal

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer


class BeamSearchLogitMetric(BaseMetric):
    def __init__(
            self,
            text_encoder,
            loss_fn: Literal["cer", "wer"],
            beam_width: int = 10,
            lm_weight: float = 0.0,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        loss_fns = {
            "cer": calc_cer,
            "wer": calc_wer,
        }

        self.text_encoder = text_encoder
        self.beam_width = beam_width
        self.lm_weight = lm_weight

        self.loss_fn = loss_fns[loss_fn]
    
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        losses = []
        
        predictions = log_probs.cpu().numpy()
        lengths = log_probs_length.detach().numpy()

        for log_prob_vecs, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_logits_decode(log_prob_vecs[:length], "beam")

            loss = self.loss_fn(target_text, pred_text)
            losses.append(loss)
        
        return sum(losses) / len(losses) * 100.0


class ArgmaxLogitMetric(BaseMetric):
    def __init__(self, text_encoder, loss_fn: Literal["cer", "wer"], *args, **kwargs):
        super().__init__(*args, **kwargs)

        loss_fns = {
            "cer": calc_cer,
            "wer": calc_wer,
        }

        self.text_encoder = text_encoder
        self.loss_fn = loss_fns[loss_fn]

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        losses = []
        
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()

        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])

            loss = self.loss_fn(target_text, pred_text)
            losses.append(loss)
        
        return sum(losses) / len(losses) * 100.0
