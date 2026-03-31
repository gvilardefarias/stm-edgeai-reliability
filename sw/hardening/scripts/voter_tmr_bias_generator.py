import os
import sys
import argparse

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
    

    # REFACTORED to use functional API properly for complex graphs like ResNet
    # We use model.layers to find the target, but we rebuild by mapping tensors.
    
    network_dict = {model.input.name: input_layer}
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        # Determine inputs
        layer_inputs = []
        # Support for functional connectivity
        inbound_nodes = layer._inbound_nodes
        for node in inbound_nodes:
            node_inputs = node.input_tensors
            if isinstance(node_inputs, list):
                for inp in node_inputs:
                    layer_inputs.append(network_dict[inp.name])
            else:
                layer_inputs.append(network_dict[node_inputs.name])
        
        if len(layer_inputs) == 1:
            layer_inputs = layer_inputs[0]

        if layer.name == target_layer_name:
            print(f"--- APPLYING BIAS TMR TO: {layer.name} ---")
            layer_weights = layer.get_weights()
            conv_weights, conv_bias = layer_weights[0], layer_weights[1]
            
            config = layer.get_config()
            config['use_bias'] = False
            config['name'] = f"{layer.name}_no_b"
            
            new_layer_node = layer.__class__.from_config(config)(layer_inputs)
            # Set weights after building
            # Find the actual layer object in the new graph to set weights
            # This is tricky in functional API. 
            # We'll use a simpler approach: create the layer object then call it.
            l_obj = layer.__class__.from_config(config)
            new_layer_node = l_obj(layer_inputs)
            l_obj.set_weights([conv_weights])
            
            # Add TMR Bias Layer
            x_out = TMRBiasLayer(original_bias=conv_bias, name=f"{layer.name}_tmr_b")(new_layer_node)
        else:
            # Clone other layers
            config = layer.get_config()
            l_obj = layer.__class__.from_config(config)
            x_out = l_obj(layer_inputs)
            l_obj.set_weights(layer.get_weights())
            
        # Update output tensor map
        # Handle multiple outputs if necessary, but here we assume single output
        if isinstance(layer.output, list):
            for i, out in enumerate(layer.output):
                network_dict[out.name] = x_out[i]
        else:
            network_dict[layer.output.name] = x_out

    tmr_model = tf.keras.Model(inputs=input_layer, outputs=network_dict[model.output.name])
    
    print("\nSaving new Bias TMR model to:", save_path)
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bias TMR hardened models.')
    parser.add_argument('--model', type=str, default='hand_posture', choices=['ign', 'hand_posture', 'miniresnet'], help='Model type')
    args = parser.parse_args()

    if args.model == 'ign':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/ign/ign_wl_24.h5"
        target = "conv2d" 
        output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/ign/HAR_bias_tmr.h5"
    elif args.model == 'hand_posture':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5"
        target = "conv2d" 
        output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/hand_posture/tmr_bias.h5"
    elif args.model == 'miniresnet':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5"
        target = "conv2_block1_1_conv" 
        output_h5 = "/home/apo/stm-edgeai-reliability/sw/hardening/hardened_models/miniresnet/tmr_bias.h5"

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    
    if os.path.exists(base_model):
        build_bias_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")