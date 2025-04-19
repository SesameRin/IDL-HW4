import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """

    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError(
                "Dimension to apply softmax to is greater than the number of dimensions in Z"
            )

        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter

        # normalize negative dim
        axis = self.dim if self.dim >= 0 else Z.ndim + self.dim

        # subtract max for numerical stability
        z_max = np.max(Z, axis=axis, keepdims=True)
        exp_Z = np.exp(Z - z_max)
        sum_exp = np.sum(exp_Z, axis=axis, keepdims=True)

        # compute softmax and cache for backward
        self.A = exp_Z / sum_exp
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass

        # determine axis and original shape
        A = self.A
        shape = A.shape
        axis = self.dim if self.dim >= 0 else len(shape) + self.dim
        C = shape[axis]

        # move softmax axis to last position
        perm = list(range(len(shape)))
        perm.pop(axis)
        perm.append(axis)
        A_perm = A.transpose(perm)
        dLdA_perm = dLdA.transpose(perm)

        # flatten to 2D: (batch, C)
        flat_shape = (-1, C)
        A_flat = A_perm.reshape(flat_shape)
        dLdA_flat = dLdA_perm.reshape(flat_shape)

        # vectorized jacobian-vector product for each row:
        # dLdZ = A * (dLdA - sum(dLdA * A, axis=1, keepdims=True))
        dot = np.sum(dLdA_flat * A_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - dot)

        # reshape back and invert transpose
        dLdZ_perm = dLdZ_flat.reshape(A_perm.shape)
        inv_perm = np.argsort(perm)
        dLdZ = dLdZ_perm.transpose(inv_perm)

        return dLdZ
