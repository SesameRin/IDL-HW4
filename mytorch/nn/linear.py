import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)

        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        # Store input for backward pass
        self.A = A

        # save original input shape
        self.input_shape = A.shape

        # flatten input to (batch_size, in_features)
        in_features = self.W.shape[1]
        A_flat = A.reshape(-1, in_features)
        # cache flattened input
        self.A_flat = A_flat

        # compute affine transform: Z_flat = A_flat @ W^T + b
        Z_flat = A_flat.dot(self.W.T) + self.b

        # reshape back to (*, out_features)
        out_features = self.W.shape[0]
        Z = Z_flat.reshape(*self.input_shape[:-1], out_features)

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        # flatten gradient to (batch_size, out_features)
        dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])

        # gradient wrt weights
        self.dLdW = dLdZ_flat.T.dot(self.A_flat)
        # gradient wrt bias
        self.dLdb = dLdZ_flat.sum(axis=0)

        # gradient wrt input (flat)
        dLdA_flat = dLdZ_flat.dot(self.W)

        # reshape gradient back to original input shape
        dLdA = dLdA_flat.reshape(self.input_shape)
        self.dLdA = dLdA

        # Return gradient of loss wrt input
        return dLdA
