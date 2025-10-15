from string import ascii_lowercase

import torch
from transformers import GPT2TokenizerFast

from src.text_encoder.basic_tokenizer import BasicTokenizer


class BPETokenizer(BasicTokenizer):
    def __init__(
            self,
            **kwargs
        ):
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        vocab = self.tokenizer.get_vocab()
        training_corpus = [ vocab.keys() ] # Should be a generator of list of texts.
        self.tokenizer = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size=2000)

        self.EMPTY_TOK = "<pad>"
        self.tokenizer.pad_token = self.EMPTY_TOK
        self.EMPTY_IND = self.tokenizer.convert_tokens_to_ids(self.EMPTY_TOK)

    def __len__(self):
        return len(self.tokenizer)

    def __getitem__(self, item: int):
        return self.tokenizer.convert_ids_to_tokens(item)

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return self.tokenizer.decode(inds, skip_special_tokens=True)
