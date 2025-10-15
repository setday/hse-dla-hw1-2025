from typing import Any, Dict, Optional, Tuple, List, Literal
from collections import defaultdict

import numpy as np
import torch

from src.text_encoder.basic_tokenizer import BasicTokenizer

# TODO add BPE, LM, Beam Search support

class LASTextEncoder:
    @staticmethod
    def _create_beams(
            tokenizer: BasicTokenizer,
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
                dp: Dict[str, float],
                next_token_probs,
                tokenizer: BasicTokenizer,

                lm_model: Optional[Any],

                alpha: float = 0.0, # LM weight
                beta: float = 0.0,  # word bonus
            ) -> Dict[str, float]:
            new_dp = defaultdict(float)
            
            for prev_path, prev_prob in dp.items():
                for next_id, next_char_prob in enumerate(next_token_probs):
                    next_char = tokenizer[next_id]

                    alpha_bonus, beta_bonus = 0.0, 0.0
                    if next_id != tokenizer.EMPTY_IND:
                        prev_path = prev_path + next_char

                        if alpha != 0.0 and lm_model is not None:
                            _, last_word = prev_path.rsplit(' ', 1) if ' ' in prev_path else (None, prev_path) # Get the last word from the hypothesis
                                                                                                            # We score only the last word since there is no strong connection between the words (this is 3/4-gram LM)
                            lm_prob = torch.tensor(lm_model.score(last_word))                    # Get the LM score for the hypothesis
                            alpha_bonus = alpha * torch.pow(10, lm_prob)
                        if beta != 0.0 and next_char == ' ':  # Apply word bonus only for the word delimiter
                            beta_bonus = beta

                    new_dp[prev_path] = prev_prob * next_char_prob + alpha_bonus + beta_bonus

            return new_dp

        def _beam_truncate_paths(
                dp: Dict[str, float],
                beam_size: int
            ) -> Dict[str, float]:
            sorted_dp = sorted(dp.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_dp[:beam_size])
        
        # --- Beam search implementation ---
        
        probs = torch.softmax(logits, dim=-1)
        beams = {'': 1.0}

        for layer_probs in probs:
            new_beams = _beam_expand_and_merge_path(beams, layer_probs, tokenizer, lm_model, alpha, beta)
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

    def __init__(
            self,
            tokenizer: BasicTokenizer,
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

        self.tokenizer = tokenizer

        self.decode_mode = decode_mode

        self.decode_lm = decode_lm

        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.tokenizer)

    def __getitem__(self, item: int):
        return self.tokenizer[item]

    def encode(self, text) -> torch.Tensor:
        return self.tokenizer.encode(text)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return self.tokenizer.decode(inds)

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
        result = []
        for ind in inds:
            if ind == self.tokenizer.EMPTY_IND:
                break
            result.append(self.tokenizer[int(ind)])
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
            beams = self._create_beams(self.tokenizer, logits, beam_width=self.beam_width, lm_model=None)
            decoded = max(beams, key=lambda x: x[1])[0].strip()
            return decoded
        elif self.decode_mode == "beam_lm":
            beams = self._create_beams(self.tokenizer, logits, beam_width=self.beam_width, lm_model=self.decode_lm)
            decoded = max(beams, key=lambda x: x[1])[0].strip()
            return decoded
        elif self.decode_mode == "beam_lm_rescore":
            beams = self._create_beams(self.tokenizer, logits, beam_width=self.beam_width, lm_model=None)
            return self._lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")
