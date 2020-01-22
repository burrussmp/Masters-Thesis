import numpy as np

X = np.random.rand(10,229,229,3)
Y = np.random.rand(10,10)
idx = int(0.8*(Y.shape[0]))-1
indices = np.arange(X.shape[0])
training_idx = np.random.choice(indices,idx,replace=False)
validation_idx = np.delete(indices,training_idx)