import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary
'''
TODO: Implement these Modules.

This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
   - Contains a stack of SelfAttentionDecoderLayers
   - Uses causal masking to prevent attending to future tokens
   - Includes optional weight tying and layer dropout features

    Key components to implement:
    1. Token Embedding Layer: Convert token IDs to vectors
    2. Positional Encoding: Add position information
    3. Decoder Stack: Process tokens sequentially
    4. Output Projection: Convert final representations to logits

    Architecture follows Pre-LN (Layer Normalization) design where:
    - Layer normalization is applied at the start of each sublayer
    - Residual connections wrap around each sublayer
    - Final layer norm is applied before output projection

    Implementation Notes:
    1. The forward pass should handle:
    - Proper masking (both padding and causal)
    - Collecting attention weights from all layers
    - Optional layer dropout during training
    
    2. The score method should:
    - Handle single token prediction
    - Not apply padding masks
    - Return only the final token's logits

2. EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
'''


## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    """
    A Pre-LN Decoder-Only Transformer model.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        """
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        """
        super().__init__()

        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Create a ModuleList of decoder layers based on the number of layers
        self.dec_layers = nn.ModuleList(
            [
                SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Create target embedding and other layers
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, num_classes)

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(
        self,
        padded_targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence logits. shape: (batch_size, seq_len, num_classes)
            running_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        """
        # DO NOT MODIFY
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # Create padding mask for padded_targets on the same device as the input
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths.to(torch.long))

        # Create causal mask to prevent attending to future tokens
        causal_mask = CausalMask(padded_targets)

        # Apply the embedding
        x = self.target_embedding(padded_targets)
        # Apply positional encoding
        x = self.positional_encoding(x)
        # Apply dropout
        x = self.dropout(x)

        # Pass through all decoder layers, save attention masks
        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue

            # Pass through decoder layer
            x, attention = layer(
                x, key_padding_mask=pad_mask_dec, attn_mask=causal_mask
            )

            # Save attention weights
            running_att[f"layer{i+1}_dec_self"] = attention

        # Apply normalization
        x = self.norm(x)
        # Linear layer (Final Projection) for next token prediction
        seq_out = self.final_linear(x)

        # Return the output sequence and running attention weights
        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        """
        Score the tokens for the decoder.
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded.
        Can only handle batch_size = 1 or batch with same lengths and no padding.
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        """
        if self.training:
            raise ValueError(
                "score method is not supported during training, use forward method instead"
            )
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the last token's logits for next token prediction
        logits = seq_out[:, -1, :]
        return logits


## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    """
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    """

    def __init__(
        self,
        input_dim,
        time_reduction,
        reduction_method,
        num_encoder_layers,
        num_encoder_heads,
        d_ff_encoder,
        num_decoder_layers,
        num_decoder_heads,
        d_ff_decoder,
        d_model,
        dropout,
        max_len,
        num_classes,
        weight_tying=False,
        layer_drop_rate=0.0,
        skip_encoder_pe=False,
        skip_decoder_pe=False,
    ):
        super().__init__()

        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        self.enc_layers = nn.ModuleList(
            [
                SelfAttentionEncoderLayer(
                    d_model, num_encoder_heads, d_ff_encoder, dropout
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(
                    d_model, num_decoder_heads, d_ff_decoder, dropout
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.source_embedding = SpeechEmbedding(
            input_dim, d_model, time_reduction, reduction_method
        )
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes), nn.LogSoftmax(dim=-1)
        )

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, padded_sources, source_lengths):
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)

        x_enc = self.dropout(x_enc)
        pad_mask_src = PadMask(x_enc, x_enc_lengths).to(x_enc.device)

        running_att = {}
        for i, layer in enumerate(self.enc_layers):
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue
            x_enc, attn = layer(x_enc, key_padding_mask=pad_mask_src)
            running_att[f"layer{i+1}_enc_self"] = attn

        x_enc = self.encoder_norm(x_enc)
        ctc_logits = self.ctc_head(x_enc.transpose(0, 1))  # (T, B, C)

        return (
            x_enc,
            pad_mask_src,
            running_att,
            {"log_probs": ctc_logits, "lengths": x_enc_lengths},
        )

    def decode(
        self, padded_targets, encoder_output, target_lengths=None, pad_mask_src=None
    ):
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths).to(
                padded_targets.device
            )

        if pad_mask_tgt is None and self.training:
            warnings.warn(
                "pad_mask_tgt is None, unless you are using the decoder as a standalone model or doing inference, you should provide target_lengths"
            )

        causal_mask = CausalMask(padded_targets).to(padded_targets.device)
        x_dec = self.target_embedding(padded_targets)

        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)

        x_dec = self.dropout(x_dec)

        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue
            x_dec, self_attn, cross_attn = layer(
                x_dec,
                encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask,
            )
            running_att[f"layer{i+1}_dec_self"] = self_attn
            running_att[f"layer{i+1}_dec_cross"] = cross_attn

        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)

        return seq_out, running_att

    def forward(
        self, padded_sources, padded_targets, source_lengths=None, target_lengths=None
    ):
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")

        encoder_output, pad_mask_src, enc_att, ctc_inputs = self.encode(
            padded_sources, source_lengths
        )
        seq_out, dec_att = self.decode(
            padded_targets, encoder_output, target_lengths, pad_mask_src
        )

        return seq_out, {**enc_att, **dec_att}, ctc_inputs

    def score(self, batch_prompts, encoder_output, pad_mask_src):
        if self.training:
            raise ValueError("score method is not supported during training")
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]

    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ) -> Tuple['EncoderDecoderTransformer', dict]:
        """
        Helper function to initialize an encoder-decoder transformer with decoder weights initialized from a pretrained decoder-only model.
        
        Args:
            decoder_checkpoint_path: Path to decoder-only transformer checkpoint
            config: Configuration dictionary for the encoder-decoder model
            
        Returns:
            model: Initialized encoder-decoder transformer
            param_info: Dictionary containing lists of named parameters {'transferred': [(name, param)], 'new': [(name, param)]}
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")

        # Create new encoder-decoder model
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']

        # Track named parameters
        transferred_params = []
        new_params = []

        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v 
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            # Store the full parameter names with their prefix
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # Transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')

        # Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")

        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )

        # Collect new parameters with their names
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))

        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0

        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable

            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


## -------------------------------------------------------------------------------------------------
## Test Cases
## -------------------------------------------------------------------------------------------------

def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(num_layers: int = 12, num_heads: int = 8, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1, max_len: int = 300, num_classes: int = 1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "__main__":
    test_decoder_only()
