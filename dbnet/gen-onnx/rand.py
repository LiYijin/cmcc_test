import numpy as np

# 创建一个 NumPy 数组，形状为 (3, 10, 10)
array = np.random.rand(3, 10, 10)

# 使用 np.expand_dims 方法在第一个维度插入一个新的维度
expanded_array = np.expand_dims(array, axis=0)

print("Original shape:", array.shape)         # 输出: (3, 10, 10)
print("Expanded shape:", expanded_array.shape) # 输出: (1, 3, 10, 10)
