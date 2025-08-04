import numpy as np

# upload tensor
tensor_flat = np.array([
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
])

# convert tensor from 16 * 4 to 4 * 4 * 4
tensor_3d = tensor_flat.reshape(4, 4, 4)

# visualize
print("Tensor shape:", tensor_3d.shape)
print("Tensor content:\n", tensor_3d)

# Save as .npy file (for use with OpenTensor)
np.save("my_custom_tensor.npy", tensor_3d)
