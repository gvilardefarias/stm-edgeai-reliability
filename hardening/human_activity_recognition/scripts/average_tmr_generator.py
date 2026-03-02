import os
import tensorflow as tf
import numpy as np

# Suppress warnings
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
            
            # 3. Set the EXACT SAME WEIGHTS for all 3 layers
            layer_1.set_weights(layer.get_weights())
            layer_2.set_weights(layer.get_weights())
            layer_3.set_weights(layer.get_weights())
            
            # 4. THE VOTER LAYER
            # We use tf.math.top_k to sort the 3 outputs and take the middle one (the median)
            # This acts as our TMR Voter without requiring complex custom layers!
            def median_voter(tensors):
                # Stack outputs: shape becomes (batch, height, width, channels, 3)
                stacked = tf.stack(tensors, axis=-1)
                # Sort the 3 values. top_k(k=2) returns the top 2 values.
                # Index 1 (the 2nd value) is guaranteed to be the median of 3 values.
                sorted_vals = tf.math.top_k(stacked, k=2).values
                return sorted_vals[..., 1]
            
            # Wrap the voter in a Lambda layer so Keras can track it
            x = tf.keras.layers.Lambda(median_voter, name=f"{layer.name}_voter")([out_1, out_2, out_3])
            print("--- VOTER LAYER ADDED ---")
            
        else:
            # For all other layers, just copy the weights and pass 'x' through
            config = layer.get_config()
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
            
    # Create the new model
    tmr_model = tf.keras.Model(inputs=input_layer, outputs=x, name="HAD_tmr")
    
    print("\nSaving new TMR model to:", save_path)
    # Save as .h5 format specifically for STM32Cube.AI
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    # Update this path to point to one of your actual trained .keras files
    base_model = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_gmp/ST_pretrainedmodel_public_dataset/WISDM/st_gmp_wl_24/st_gmp_wl_24.keras"
    
    # We will pick a layer to triplicate. Let's assume there is a layer named 'conv2d_1'
    # You can change this to any layer name Gustavo tells you to harden!
    target = "conv2d_bias" 
    
    output_h5 = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_gmp/ST_pretrainedmodel_public_dataset/WISDM/st_gmp_wl_24/HAR_tmr.h5"
    
    build_tmr_model(base_model, target, output_h5)