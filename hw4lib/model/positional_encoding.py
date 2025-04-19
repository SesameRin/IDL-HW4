import torch
from torch import nn
import math

"""
TODO: Implement this Module.

Specification:
- Module should add positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix should have shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values should be on same device as input tensor
- Should handle any sequence length up to max_len
- Should raise error if input sequence length exceeds max_len
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.

        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.

        Side Effects:
            - Initializes the positional encoding buffer 'pe'
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        # Build a matrix of positions (T x 1)
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(
            1
        )  # shape (max_len, 1)
        # Compute the div term for each even dimension index
        # shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )  # shape (d_model/2,)
        # Initialize pe tensor of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Apply sine to even indices: pe[pos, 2i] = sin(pos * div_term[i])
        pe[:, 0::2] = torch.sin(positions * div_term)
        # Apply cosine to odd indices: pe[pos, 2i+1] = cos(pos * div_term[i])
        pe[:, 1::2] = torch.cos(positions * div_term)
        # Add batch dimension for broadcasting: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register as buffer so it's saved with the model but not learnable
        self.register_buffer("pe", pe)
        # Store max length for forward check
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum supported length
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the maximum length {self.max_len}"
            )
        # Step 3: Add positional encodings to input embeddings
        # Slice the positional matrix up to seq_len and add
        # pe: (1, max_len, d_model) -> (1, seq_len, d_model)
        pe_slice = self.pe[:, :seq_len, :]
        # Return sum: broadcasting over batch dimension
        return x + pe_slice
