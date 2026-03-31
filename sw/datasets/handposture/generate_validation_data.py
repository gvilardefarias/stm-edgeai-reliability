import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_file = '/home/apo/stm-edgeai-reliability/sw/datasets/handposture/hand_val_images.npy'
base_model_path = '/home/apo/stm-edgeai-reliability/sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5'

val_x_path = '/home/apo/stm-edgeai-reliability/sw/datasets/handposture/val_x.npy'
val_y_path = '/home/apo/stm-edgeai-reliability/sw/datasets/handposture/val_y.npy'

print(f"Loading inputs from {input_file}...")
x = np.load(input_file).astype(np.float32)
np.save(val_x_path, x)
print(f"Saved ST Edge AI inputs to {val_x_path} with shape {x.shape}")

print(f"Loading Keras model {base_model_path} to generate expected outputs...")
model = tf.keras.models.load_model(base_model_path)
y = model.predict(x, batch_size=32)

np.save(val_y_path, y.astype(np.float32))
print(f"Saved ST Edge AI expected outputs to {val_y_path} with shape {y.shape}")
