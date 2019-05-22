import numpy as np

x = np.array([[1, 2], [4, 5]], np.int32)
y = np.array([[1, 2], [4, 5],  [7, 8]], np.int32)
y_t = y.T

x_dim = x.shape[1];
y_t_dim = y_t.shape[0];

if x_dim != y_t_dim:
    print("matrix shapes do not match")
else:
    print(x.dot(y_t))

