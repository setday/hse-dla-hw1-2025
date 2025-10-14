import re
from string import ascii_lowercase
from typing import Any, Dict, Optional, Tuple, List, Literal
from collections import defaultdict

import numpy as np
import torch

# TODO add BPE, LM, Beam Search support

class CTCTextEncoder:
    @staticmethod
    def _create_beams(
            ind2char: Dict[int, str],
            logits: torch.Tensor,
            beam_width: int = 10,
            lm_model: Optional[Any] = None,
            alpha: float = 1.23,
            beta: float = -0.26
        ) -> List[Tuple[str, float]]:
        """
        Perform beam search decoding (no LM).
        
        Args:
            ind2char (Dict[int, str]): Mapping from indices to characters
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            beam_width (int): Number of beams to keep during decoding
            lm_model (Any): External language model with a 'score' method
            alpha (float): Language model weight
            beta (float): Word bonus

        Returns:
            (str) the best decoded transcript as a string.

        Note:
            This implementation is based on my AITH homework (whole implementation is mine, except the interfaces).
        """ 

        # --- Auxiliary functions for beam search ---

        def _beam_expand_and_merge_path(
                dp: Dict[Tuple[str, int], float],
                next_token_probs,
                ind2char: Dict[int, str],

                lm_model: Optional[Any],

                alpha: float = 0.0, # LM weight
                beta: float = 0.0,  # word bonus
            ) -> Dict[Tuple[str, int], float]:
            new_dp = defaultdict(float)
            
            for (prev_path, last_ind), prev_prob in dp.items():
                for next_id, next_char_prob in enumerate(next_token_probs):
                    next_char = ind2char[next_id]

                    alpha_bonus, beta_bonus = 0.0, 0.0
                    if next_id != last_ind and next_id != CTCTextEncoder.EMPTY_IND:
                        new_path = prev_path + next_char

                        if alpha != 0.0 and lm_model is not None:
                            _, last_word = new_path.rsplit(' ', 1) if ' ' in new_path else (None, new_path) # Get the last word from the hypothesis
                                                                                                            # We score only the last word since there is no strong connection between the words (this is 3/4-gram LM)
                            lm_prob = torch.tensor(lm_model.score(last_word))                    # Get the LM score for the hypothesis
                            alpha_bonus = alpha * torch.pow(10, lm_prob)
                        if beta != 0.0 and next_char == ' ':  # Apply word bonus only for the word delimiter
                            beta_bonus = beta
                    else:
                        new_path = prev_path

                    new_dp[(new_path, next_id)] += prev_prob * next_char_prob + alpha_bonus + beta_bonus

            return new_dp

        def _beam_truncate_paths(
                dp: Dict[Tuple[str, int], float],
                beam_size: int
            ) -> Dict[Tuple[str, int], float]:
            sorted_dp = sorted(dp.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_dp[:beam_size])
        
        # --- Beam search implementation ---
        
        probs = torch.softmax(logits, dim=-1)
        beams = {('', CTCTextEncoder.EMPTY_IND): 1.0}

        for layer_probs in probs:
            new_beams = _beam_expand_and_merge_path(beams, layer_probs, ind2char, lm_model, alpha, beta)
            beams = _beam_truncate_paths(new_beams, beam_width)

        beams = [(hyp, np.log10(prob + 1e-10)) for (hyp, _), prob in beams.items()]

        return beams
        
    def _lm_rescore(self, beams: List[Tuple[str, float]], alpha: float = 1.0) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
            lm_model (Any): External language model with a 'score' method
            alpha (float): Weight for the LM score during rescoring
        
        Returns:
            str: Best rescored transcript

        Note:
            This implementation is based on my AITH homework (whole implementation is mine, except the interfaces).
        """
        if self.decode_lm is None:
            raise ValueError("Language model not set. Please set it using 'set_language_model' method before LM rescoring.")

        rescored_beams = []
        for hypothesis, log_prob in beams:
            lm_prob = self.decode_lm.score(hypothesis)
            rescored_prob = log_prob + alpha * lm_prob
            rescored_beams.append((hypothesis, rescored_prob))
            
        best_hypothesis = max(rescored_beams, key=lambda x: x[1])[0]

        return best_hypothesis.strip()

    EMPTY_TOK = ""
    EMPTY_IND = 0

    WORD_DELIMITER_TOK = None  # Set to None for character-level, or to specific index for word-level (e.g., space)
    WORD_DELIMITER_IND = None

    def __init__(
            self,
            alphabet=None,
            decode_mode: Literal["greedy", "beam", "beam_lm", "beam_lm_rescore"] = "greedy",
            decode_lm: Optional[Any] = None,
            beam_width: Optional[int] = None,
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
            **kwargs
        ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
            decode_lm (Any): External language model with a 'score' method for decoding
            beam_width (int): Beam width for beam search decoding
            alpha (float): LM weight for beam search decoding
            beta (float): Word bonus for beam search decoding
        """

        assert decode_mode == "greedy" or beam_width is not None, \
            "Beam width must be specified for beam search decoding."
        assert decode_mode == "greedy" or (alpha is not None and beta is not None), \
            "Alpha and beta must be specified for beam search decoding."
        assert decode_mode in ["greedy", "beam"] or decode_lm is not None, \
            "Language model must be provided for LM-based decoding."

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decode_mode = decode_mode

        self.decode_lm = decode_lm

        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Greedy CTC decoding: remove consecutive duplicates and blanks.
        Args:
            inds (list or tensor): sequence of token indices
        Returns:
            Decoded string
        """
        if isinstance(inds, torch.Tensor):
            inds = inds.cpu().numpy().tolist()
        prev = None
        result = []
        for ind in inds:
            if ind != self.EMPTY_IND and ind != prev:
                result.append(self.ind2char[int(ind)])
            prev = ind
        return "".join(result).strip()
    
    def logits_decode(self, logits: torch.Tensor) -> str:
        """
        CTC beam search decoding with optional LM rescoring.
        
        Args:
            logits (torch.Tensor): Logits from the model (T, V), where
                T - number of time steps and
                V - vocabulary size
            beam_width (int): Number of beams to keep during decoding
        
        Returns:
            str: Best decoded transcript after beam search and optional LM rescoring.

        Note:
            This implementation is based on my AITH homework (whole implementation is mine, except the interfaces).
        """

        if self.decode_mode == "greedy":
            maxes = torch.argmax(logits, dim=-1)
            return self.ctc_decode(maxes)
        elif self.decode_mode == "beam":
            beams = self._create_beams(self.ind2char, logits, beam_width=self.beam_width, lm_model=None)
            decoded = max(beams, key=lambda x: x[1])[0].strip()
            return decoded
        elif self.decode_mode == "beam_lm":
            beams = self._create_beams(self.ind2char, logits, beam_width=self.beam_width, lm_model=self.decode_lm)
            decoded = max(beams, key=lambda x: x[1])[0].strip()
            return decoded
        elif self.decode_mode == "beam_lm_rescore":
            beams = self._create_beams(self.ind2char, logits, beam_width=self.beam_width, lm_model=None)
            return self._lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
