import numpy as np
import torch

# seed() 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
np.random.seed(2)

T = 20
L = 1000
N = 100

# np.empty: empty, unlike zeros, does not set the array values to zero, and may therefore be marginally faster. 
# On the other hand, it requires the user to manually set all the values in the array, and should be used with caution.
x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))
