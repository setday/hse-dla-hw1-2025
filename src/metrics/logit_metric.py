from typing import List, Literal, Optional

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer


class LogitMetric(BaseMetric):
    def __init__(
            self,
            text_encoder,
            loss_fn: Literal["cer", "wer"],
            decode_mode: Literal["beam", "greedy", "beam_lm", "beam_lm_rescore"] = "beam_lm",
            beam_width: Optional[int] = 10,
            lm_weight: Optional[float] = 1.23,
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
        self.decode_mode = decode_mode

        self.loss_fn = loss_fns[loss_fn]

    def measure_single(self, log_probs: Tensor, log_probs_length: Tensor, text: str):
        log_probs = log_probs[:log_probs_length.item()]

        target_text = self.text_encoder.normalize_text(text)
        pred_text = self.text_encoder.logits_decode(log_probs, self.decode_mode)

        loss = self.loss_fn(target_text, pred_text)
        return loss        
    
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        predictions = log_probs.cpu()
        lengths = log_probs_length.detach().cpu()

        losses = [
            self.measure_single(log_prob_vecs, length, target_text)
            for log_prob_vecs, length, target_text in zip(predictions, lengths, text)
        ]
        
        return sum(losses) / len(losses) * 100.0
