import sys
import os
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_layers')))
from median_voter_layer import MedianVoterLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_tmr_model(original_model_path, target_layer_name, save_path):
    print(f"Loading original model: {original_model_path}")
    # Load your trained model (can be .keras or .h5)
    model = tf.keras.models.load_model(original_model_path)
    
    # We will use the Functional API to rebuild the model
    # and inject our TMR logic at the target layer
    
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = input_layer
    
    for layer in model.layers:
        # Skip the input layer as we already created it
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        if layer.name == target_layer_name:
            print(f"--- TRIPLICATING LAYER: {target_layer_name} ---")
            
            # 1. Clone the layer 3 times
            # We get the config to ensure strides, padding, etc., are identical
            config = layer.get_config()
            config['name'] = layer.name + "_tmr1"
            layer_1 = layer.__class__.from_config(config)
            
            config['name'] = layer.name + "_tmr2"
            layer_2 = layer.__class__.from_config(config)
            
            config['name'] = layer.name + "_tmr3"
            layer_3 = layer.__class__.from_config(config)
            
            # 2. Pass the current input 'x' through all 3 branches
            out_1 = layer_1(x)
            out_2 = layer_2(x)
            out_3 = layer_3(x)
            
            # 3. Set slightly different weights for all 3 layers to avoid deduplication
            import numpy as np
            weights = layer.get_weights()
            for i, l in enumerate([layer_1, layer_2, layer_3]):
                epsilon = (i + 1) * 1e-6
                modified_weights = []
                for w in weights:
                    if np.issubdtype(w.dtype, np.floating):
                        modified_weights.append((w.copy() + epsilon).astype(np.float32))
                    else:
                        modified_weights.append(w.copy())
                l.set_weights(modified_weights)
            
            # 4. THE VOTER LAYER
            # We use tf.math.top_k to sort the 3 outputs and take the middle one (the median)
            # This acts as our TMR Voter without requiring complex custom layers!
            x = MedianVoterLayer(name=f"{layer.name}_voter")([out_1, out_2, out_3])
            print("--- MEDIAN VOTER LAYER ADDED ---")
            
        else:
            # For all other layers, just copy the weights and pass 'x' through
            config = layer.get_config()
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
            
    # Create the new model
    tmr_model = tf.keras.Model(inputs=input_layer, outputs=x, name="HAR_tmr")
    
    print("\nSaving new TMR model to:", save_path)
    # Save as .h5 format specifically for STM32Cube.AI
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    # Update this path to point to one of your actual trained .keras files
    base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/gmp/gmp_wl_24.h5"
    
    # We will pick a layer to triplicate. Let's assume there is a layer named 'conv2d_1'
    # You can change this to any layer name Gustavo tells you to harden!
    target = "conv2d_1" 
    
    output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/gmp/HAR_avg_tmr.h5"
    
    if os.path.exists(base_model):
        build_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")