from src.metrics.logit_metric import ArgmaxLogitMetric, BeamSearchLogitMetric


class BeamsearchWERMetric(BeamSearchLogitMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(text_encoder, "wer", *args, **kwargs)

class ArgmaxWERMetric(ArgmaxLogitMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(text_encoder, "wer", *args, **kwargs)
