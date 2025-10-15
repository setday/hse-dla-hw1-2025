import re

import torch

class BasicTokenizer:
    EMPTY_TOK = ""
    EMPTY_IND = 0

    def __init__(
            self,
            **kwargs
        ):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError

    def encode(self, text) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, inds) -> str:
        """
        Raw decoding.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        raise NotImplementedError

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
