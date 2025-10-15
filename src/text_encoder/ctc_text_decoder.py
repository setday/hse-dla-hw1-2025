from typing import List

from torch import Tensor

from src.metrics.base_metric import BaseMetric


class TextDecoder(BaseMetric):
    def __init__(
            self,
            text_encoder,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.text_encoder = text_encoder

    def assemble_predictions(self, log_probs: Tensor, log_probs_length: Tensor) -> List[str]:
        return [
            self.text_encoder.logits_decode(pred[: length.item()])
            for pred, length in zip(log_probs.cpu(), log_probs_length.detach().cpu())
        ]

    def transform_targets(self, text: List[str]) -> List[str]:
        return [self.text_encoder.tokenizer.normalize_text(t) for t in text]

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> dict:
        preds = self.assemble_predictions(log_probs, log_probs_length)
        targets = self.transform_targets(text)
        
        return {"text_predicted": preds, "text_target": targets}
