from typing import List, Literal

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.text_encoder.ctc_text_decoder import TextDecoder
from src.metrics.utils import calc_cer, calc_wer


class LogitMetric(BaseMetric):
    def __init__(
            self,
            loss_fn: Literal["cer", "wer"],
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.sim_fn = {
            "cer": calc_cer,
            "wer": calc_wer,
        }[loss_fn]

    def __call__(
        self, text_predicted: List[str], text_target: List[str], **kwargs
    ):
        losses = [self.sim_fn(tgt, pred) for pred, tgt in zip(text_predicted, text_target)]
        return sum(losses) / len(losses) * 100.0
