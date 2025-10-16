import pickle
import numpy as np

data_loaded = np.load("gmp/test.pkl", allow_pickle=True)
data_loaded = data_loaded['acc']

out_data = np.array([])
for array in data_loaded:
    out_data = np.append(out_data, array)

np.save("gmp/test.npy", out_data)
