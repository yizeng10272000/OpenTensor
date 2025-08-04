import numpy as np

# 你的 tensor（4×16）输入数据
tensor_flat = np.array([
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
])

# 将每一行 reshape 成 4×4，然后 stack 成 4×4×4 张量
tensor_3d = tensor_flat.reshape(4, 4, 4)

# 可视化（可选）
print("Tensor shape:", tensor_3d.shape)
print("Tensor content:\n", tensor_3d)

# 保存成 .npy 文件（供 OpenTensor 使用）
np.save("my_custom_tensor.npy", tensor_3d)
