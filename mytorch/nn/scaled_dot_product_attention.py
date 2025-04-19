import numpy as np
from .activation import Softmax


class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """

    def __init__(self):
        """
        Initialize the ScaledDotProductAttention class.
        """
        self.eps = 1e10  # DO NOT MODIFY
        # softmax over the last dimension (source sequence length S)
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: shape (N,...,H,L,E)
        :param K: shape (N,...,H,S,E)
        :param V: shape (N,...,H,S,Ev)
        :param mask: broadcastable to (N,...,H,L,S), True = ignore
        :return: shape (N,...,H,L,Ev)
        """
        # 1) raw dot product: (N,...,H,L,S)
        #    K^T over the last two dims
        scaled_dot = np.matmul(Q, K.swapaxes(-2, -1))

        # 2) scale by sqrt(d_k)
        d_k = Q.shape[-1]
        scaled_dot = scaled_dot / np.sqrt(d_k)

        # 3) apply mask if provided
        if mask is not None:
            # wherever mask==True, subtract a large number
            scaled_dot = np.where(mask, scaled_dot - self.eps, scaled_dot)

        # 4) softmax over S dimension
        #    store for backward
        self.attention_scores = self.softmax.forward(scaled_dot)

        # 5) weighted sum with V -> (N,...,H,L,Ev)
        output = np.matmul(self.attention_scores, V)

        # cache inputs for backward
        self.Q, self.K, self.V, self.mask = Q, K, V, mask
        return output

    def backward(self, d_output):
        """
        :param d_output: shape (N,...,H,L,Ev)
        :return: dQ, dK, dV with shapes matching Q,K,V
        """
        Q, K, V, mask = self.Q, self.K, self.V, self.mask
        scores = self.attention_scores
        d_k = Q.shape[-1]

        # 1) grad wrt V:  scores^T @ d_output  -> (N,...,H,S,Ev)
        d_V = np.matmul(scores.swapaxes(-2, -1), d_output)

        # 2) grad wrt attention scores:  d_output @ V^T -> (N,...,H,L,S)
        d_scores = np.matmul(d_output, V.swapaxes(-2, -1))

        # 3) back through softmax: get grad wrt scaled_dot
        d_scaled = self.softmax.backward(d_scores)

        # 4) undo the scaling:  d_raw = d_scaled / sqrt(d_k)
        d_raw = d_scaled / np.sqrt(d_k)

        # 5a) grad wrt Q:  d_raw @ K -> (N,...,H,L,E)
        d_Q = np.matmul(d_raw, K)
        # 5b) grad wrt K:  d_raw^T @ Q -> (N,...,H,S,E)
        d_K = np.matmul(d_raw.swapaxes(-2, -1), Q)

        return d_Q, d_K, d_V
