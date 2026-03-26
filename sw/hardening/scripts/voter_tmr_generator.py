import os
# Force TensorFlow to use Keras 2 (Compatible with ST Edge AI v1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sys
# Make sure python can find your custom layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_layers')))
from majority_voter_layer import MajorityVoterLayer

def build_tmr_model(original_model_path, target_layer_name, save_path):
    print(f"Loading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = input_layer
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        if layer.name == target_layer_name:
            print(f"--- TRIPLICATING LAYER: {target_layer_name} ---")
            
            # 1. Clone the layer 3 times
            config = layer.get_config()
            layers = []
            for i in range(1, 4):
                config['name'] = f"{layer.name}_tmr{i}"
                l = layer.__class__.from_config(config)
                layers.append(l)
            
            # 2. Pass current input through all 3 branches
            out_1 = layers[0](x)
            out_2 = layers[1](x)
            out_3 = layers[2](x)
            
            # 3. Set slightly different weights to prevent ST Edge AI deduplication
            weights = layer.get_weights()
            for i, l in enumerate(layers):
                epsilon = (i + 1) * 1e-6
                modified_weights = []
                for w in weights:
                    if np.issubdtype(w.dtype, np.floating):
                        modified_weights.append((w.copy() + epsilon).astype(np.float32))
                    else:
                        modified_weights.append(w.copy())
                l.set_weights(modified_weights)
            
            # 4. MAJORITY VOTER LAYER
            x = MajorityVoterLayer(name=f"{layer.name}_voter")([out_1, out_2, out_3])
            print("--- MAJORITY VOTER LAYER ADDED ---")
            
        else:
            config = layer.get_config()
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
            
    tmr_model = tf.keras.Model(inputs=input_layer, outputs=x, name="HandPosture_TMR")
    print("\nSaving new TMR model to:", save_path)
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/gmp/gmp_wl_24.h5"
    target = "conv2d" 
    output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/gmp/HAR_tmr_voter.h5"
    
    if os.path.exists(base_model):
        build_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")