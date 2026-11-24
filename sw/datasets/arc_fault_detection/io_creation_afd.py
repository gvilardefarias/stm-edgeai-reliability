import numpy as np
from arc_fault_detection.tf.src.datasets.load_datasets import load_file_from_zip, preprocess_data

#the script must run on the arc_fault_detection dir on stm32-modelzoo-services github. the two .csv file used are found in the dateset dir on arc_fault_detection dir from stm32-modelzoo-services page 

shape = (1, 512, 1)                                   # input_shape del modello
X_tr, y_tr = load_file_from_zip("arc_fault_detection/datasets/afd_test_bench/Arc_and_Normal_train.csv")
_, _, params = preprocess_data(X_tr, y_tr, shape, True, True, True)   # calcola (mean, std) dal train

X, y = load_file_from_zip(r"C:\ST\workspace\arc_fault_detection\afd_test_bench\Arc_and_Normal_test.csv")
X, y = preprocess_data(X, y, shape, True, True, params)
np.save("inputs.npy", X.reshape(len(X), -1))                          # una riga per campione

y = y.reshape(-1)                              # (1000, 1) -> (1000,)
y_onehot = np.eye(2, dtype=np.float32)[y]      # (1000, 2)  0->[1,0], 1->[0,1]
np.save("outputs.npy", y_onehot)