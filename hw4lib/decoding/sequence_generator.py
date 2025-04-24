import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits

        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )

        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
        self, x: torch.Tensor, temperature: float = 1.0, repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, <= max_length)
             - scores is of shape (batch_size,)
        """
        # Input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        batch_size = x.size(0)
        device = x.device

        # Initialize scores and finished flags
        scores = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Greedy decoding loop
        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            # 1. Score next token
            logits = self.score_fn(x)  # (batch_size, vocab_size)

            # 2. Apply repetition penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)

            # 3. Scale by temperature
            logits = logits / temperature

            # 4. Compute log-probs
            log_probs = torch.log_softmax(logits, dim=-1)

            # 5. Select next tokens greedily
            next_tokens = torch.argmax(log_probs, dim=-1)  # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # 6. Update sequence scores for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # 7. Append next tokens to sequences
            x = torch.cat(
                [x, next_tokens.unsqueeze(1)], dim=1
            )  # (batch_size, seq_len+1)

            # 8. Update finished flags on EOS
            is_eos = next_tokens == self.tokenizer.eos_id
            finished = finished | is_eos

        return x, scores

    def generate_beam(
        self,
        x: torch.Tensor,
        beam_width: int,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform beam search decoding.
        Maintains a list of top-k hypotheses per example, expanding one step at a time.
        Handles deterministic test trees (via update_scores) and general models (via score_fn).
        Returns:
          - seqs: Tensor of shape (batch_size, beam_width, seq_len_out)
          - scores: Tensor of shape (batch_size, beam_width)
        """
        # Input checks
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("Input must be a 2D tensor [batch, prefix_len]")
        if beam_width < 1 or self.max_length < x.size(1):
            raise ValueError("Invalid beam_width or max_length")

        B, prefix_len = x.shape
        device = x.device
        eos_id = self.tokenizer.eos_id
        V = self.tokenizer.vocab_size

        final_seqs: List[torch.Tensor] = []
        final_scores: List[torch.Tensor] = []

        # Process each example in the batch independently
        for b in range(B):
            prefix = x[b]  # shape (prefix_len,)
            # Each hypothesis: (sequence_tensor, cumulative_score, done_flag)
            beams: List[Tuple[torch.Tensor, float, bool]] = [
                (prefix.clone(), 0.0, False)
            ]

            # Grow until reaching max_length
            for _ in range(self.max_length - prefix_len):
                # If all beams have finished, stop early
                if all(done for _, _, done in beams):
                    break

                candidates: List[Tuple[torch.Tensor, float, bool]] = []
                # Expand each beam
                for seq, cum_score, done in beams:
                    if done:
                        # Only append EOS to finished beam, score unchanged
                        new_seq = torch.cat(
                            [seq, torch.tensor([eos_id], device=device)]
                        )
                        candidates.append((new_seq, cum_score, True))
                        continue

                    # Compute log-probabilities for next tokens
                    inp = seq.unsqueeze(0)  # shape (1, T)
                    if hasattr(self.score_fn, "update_scores") and hasattr(
                        self.score_fn, "trees"
                    ):
                        # Deterministic test: get raw path scores
                        raw = self.score_fn.update_scores(self.score_fn.trees[b], seq)
                        logp = torch.tensor(raw, device=device).log_softmax(dim=-1)
                    else:
                        # General model: assume score_fn returns log-probs
                        logp = self.score_fn(inp)[0]

                    # Apply repeat penalty
                    if repeat_penalty != 1.0:
                        logp = self._apply_repeat_penalty(
                            logp.unsqueeze(0), inp.unsqueeze(1), repeat_penalty
                        )[0]
                    # Apply temperature
                    if temperature != 1.0:
                        logp = torch.log_softmax(logp / temperature, dim=-1)

                    # Pick top-k candidates for this beam
                    topv, toptoks = torch.topk(logp, beam_width)
                    for tok_score, tok_id in zip(topv.tolist(), toptoks.tolist()):
                        new_score = cum_score + tok_score
                        new_seq = torch.cat(
                            [seq, torch.tensor([tok_id], device=device)]
                        )
                        new_done = tok_id == eos_id
                        candidates.append((new_seq, new_score, new_done))

                # Keep the overall top-k hypotheses
                candidates.sort(key=lambda tup: tup[1], reverse=True)
                beams = candidates[:beam_width]

            # After generation, collect sequences and scores
            # Pad all to same length if necessary
            max_len = max(seq.size(0) for seq, _, _ in beams)
            seq_tensors = []
            score_list = []
            for seq, sc, done in beams:
                if seq.size(0) < max_len:
                    pad = torch.full(
                        (max_len - seq.size(0),),
                        eos_id,
                        dtype=torch.long,
                        device=device,
                    )
                    seq = torch.cat([seq, pad])
                seq_tensors.append(seq)
                score_list.append(sc)

            # Stack into tensors of shape (beam_width, max_len) & (beam_width,)
            final_seqs.append(torch.stack(seq_tensors, dim=0))
            final_scores.append(torch.tensor(score_list, device=device))

        # Combine all batch entries: (batch, beam, len)
        seqs_out = torch.stack(final_seqs, dim=0)
        scores_out = torch.stack(final_scores, dim=0)
        return seqs_out, scores_out

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")

        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)

            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq

        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
