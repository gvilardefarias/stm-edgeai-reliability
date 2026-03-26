import os
import sys

# Suppress warnings (must be set before importing TF)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Import the custom layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_layers')))
from tmr_bias_layer import TMRBiasLayer

def build_bias_tmr_model(original_model_path, target_layer_name, save_path):
    print(f"Loading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = input_layer
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        if layer.name == target_layer_name:
            print(f"--- APPLYING BIAS TMR TO: {layer.name} ---")
            
            # 1. Extract weights and biases from the original layer
            layer_weights = layer.get_weights()
            if len(layer_weights) < 2:
                print(f"Error: Layer {layer.name} does not have a bias.")
                return
            
            conv_weights = layer_weights[0]
            conv_bias = layer_weights[1]
            
            # 2. Rebuild the Conv2D layer WITHOUT the bias
            # Get fresh config for THIS specific layer
            layer_config = layer.get_config()
            layer_config['use_bias'] = False
            
            # Use the safe _no_b suffix to avoid ST Edge AI string matching bugs
            layer_config['name'] = f"{layer.name}_no_b" 
            conv_nobias_layer = layer.__class__.from_config(layer_config)
            
            # 3. Pass input through the bias-less convolution and set its weights
            x = conv_nobias_layer(x)
            conv_nobias_layer.set_weights([conv_weights])
            
            # 4. Apply the Custom TMR Bias Voter Layer
            # Use the safe _tmr_b suffix to avoid ST Edge AI string matching bugs
            tmr_bias_layer = TMRBiasLayer(original_bias=conv_bias, name=f"{layer.name}_tmr_b")
            x = tmr_bias_layer(x)
            
            print("--- BIAS SEPARATED AND VOTER ADDED ---")
            
        else:
            # For all other layers, just copy the config and weights directly
            other_config = layer.get_config()
            new_layer = layer.__class__.from_config(other_config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
            
    tmr_model = tf.keras.Model(inputs=input_layer, outputs=x)
    print("\nSaving new Bias TMR model to:", save_path)
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    # Ensure these paths match your environment
    base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/gmp/gmp_wl_24.h5"
    target = "conv2d" 
    
    # Save to the hardened_models directory matching your validation command
    output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/gmp/HAR_bias_tmr_1.h5"
    
    if os.path.exists(base_model):
        build_bias_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")