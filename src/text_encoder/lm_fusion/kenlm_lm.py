import kenlm


class KenLMLanguageModel:
    def __init__(self, lm_path: str, **kwargs):
        self.model = kenlm.Model(lm_path)

    def score(self, text: str) -> float:
        return self.model.score(text, eos=False)
