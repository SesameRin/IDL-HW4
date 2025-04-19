from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np


class MultiHeadAttention:
    """
    Multi Head Attention
    """

    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()

        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        # project inputs
        q = self.q_proj.forward(query)  # (N, L, E)
        k = self.k_proj.forward(key)  # (N, S, E)
        v = self.v_proj.forward(value)  # (N, S, E)

        # split heads
        qh = self._split_heads(q)  # (N, H, L, E/H)
        kh = self._split_heads(k)  # (N, H, S, E/H)
        vh = self._split_heads(v)  # (N, H, S, E/H)

        # merge masks
        mask = self._merge_masks(key_padding_mask, attn_mask)  # (N, H, L, S)

        # apply attention per head
        attn_outputs = self.attention.forward(qh, kh, vh, mask=mask)  # (N, H, L, E/H)

        # concat heads
        concat = self._concat_heads(attn_outputs)  # (N, L, E)

        # final linear projection
        output = self.out_proj.forward(concat)  # (N, L, E)

        # cache for backward
        self._cache = (query, key, value, q, k, v, qh, kh, vh, attn_outputs, concat)
        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """
        (query, key, value, q, k, v, qh, kh, vh, attn_outputs, concat) = self._cache

        # backprop through out_proj
        d_concat = self.out_proj.backward(d_output)  # (N, L, E)

        # split gradient to heads
        d_attn_outputs = self._split_heads(d_concat)  # (N, H, L, E/H)

        # backprop through attention
        d_qh, d_kh, d_vh = self.attention.backward(d_attn_outputs)

        # merge heads back
        d_q = self._concat_heads(d_qh)  # (N, L, E)
        d_k = self._concat_heads(d_kh)  # (N, S, E)
        d_v = self._concat_heads(d_vh)  # (N, S, E)

        # backprop through input projections
        d_query = self.q_proj.backward(d_q)  # (N, L, E)
        d_key = self.k_proj.backward(d_k)  # (N, S, E)
        d_value = self.v_proj.backward(d_v)  # (N, S, E)

        return d_query, d_key, d_value

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        N = None if key_padding_mask is None else key_padding_mask.shape[0]
        H = self.num_heads

        # key padding mask -> (N, 1, 1, S)
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]
        else:
            key_mask = None

        # attn mask -> (1, 1, L, S)
        if attn_mask is not None:
            attention_mask = attn_mask[None, None, :, :]
        else:
            attention_mask = None

        # combine
        if key_mask is not None and attention_mask is not None:
            combined = np.logical_or(key_mask, attention_mask)
        elif key_mask is not None:
            combined = np.broadcast_to(
                key_mask, (key_mask.shape[0], H, key_mask.shape[2], key_mask.shape[3])
            )
        elif attention_mask is not None:
            combined = np.broadcast_to(
                attention_mask,
                (
                    self._cache[0].shape[0],
                    H,
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                ),
            )
        else:
            combined = None

        return combined

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        N, L, E = x.shape
        H = self.num_heads
        d_k = E // H
        # reshape: (N, L, H, d_k)
        x = x.reshape(N, L, H, d_k)
        # transpose: (N, H, L, d_k)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        N, H, L, d_k = x.shape
        # transpose: (N, L, H, d_k)
        x = x.transpose(0, 2, 1, 3)
        # reshape: (N, L, H*d_k)
        x = x.reshape(N, L, H * d_k)
        return x
