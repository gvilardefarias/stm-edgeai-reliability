import sys
import os
import tensorflow as tf
import numpy as np
import argparse

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import local custom layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_layers')))
from median_voter_layer import MedianVoterLayer

def build_tmr_model(original_model_path, target_layer_name, save_path):
    print(f"Loading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    network_dict = {model.input.name: input_layer}
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        # Determine inputs using functional nodes
        layer_inputs = []
        for node in layer._inbound_nodes:
            node_inputs = node.input_tensors
            if isinstance(node_inputs, list):
                for inp in node_inputs:
                    layer_inputs.append(network_dict[inp.name])
            else:
                layer_inputs.append(network_dict[node_inputs.name])
        
        if len(layer_inputs) == 1:
            layer_inputs = layer_inputs[0]

        if layer.name == target_layer_name:
            print(f"--- TRIPLICATING LAYER: {target_layer_name} ---")
            
            # 1. Clone the layer 3 times
            config = layer.get_config()
            weights = layer.get_weights()
            
            outs = []
            for i in range(3):
                config['name'] = f"{layer.name}_tmr{i+1}"
                l_obj = layer.__class__.from_config(config)
                out = l_obj(layer_inputs)
                
                # Apply epsilon to weights to bypass deduplication
                epsilon = (i + 1) * 1e-6
                modified_weights = []
                for w in weights:
                    if np.issubdtype(w.dtype, np.floating):
                        modified_weights.append((w.copy() + epsilon).astype(np.float32))
                    else:
                        modified_weights.append(w.copy())
                l_obj.set_weights(modified_weights)
                outs.append(out)
            
            # 2. Add Median Voter
            x_out = MedianVoterLayer(name=f"{layer.name}_voter")(outs)
            print("--- MEDIAN VOTER LAYER ADDED ---")
            
        else:
            # Copy other layers
            config = layer.get_config()
            l_obj = layer.__class__.from_config(config)
            x_out = l_obj(layer_inputs)
            l_obj.set_weights(layer.get_weights())
            
        # Map output tensors
        if isinstance(layer.output, list):
            for i, out in enumerate(layer.output):
                network_dict[out.name] = x_out[i]
        else:
            network_dict[layer.output.name] = x_out

    tmr_model = tf.keras.Model(inputs=input_layer, outputs=network_dict[model.output.name], name="HAR_tmr")
    
    print("\nSaving new TMR model to:", save_path)
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Average/Median TMR hardened models.')
    parser.add_argument('--model', type=str, default='hand_posture', choices=['ign', 'hand_posture', 'miniresnet', 'gmp'], help='Model type')
    args = parser.parse_args()

    # Dynamic project root detection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    if args.model == 'ign':
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/ign/ign_wl_24.h5")
        target = "conv2d" 
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/ign/avg_tmr.h5")
    elif args.model == 'hand_posture':
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5")
        target = "conv2d" 
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/hand_posture/avg_tmr.h5")
    elif args.model == 'miniresnet':
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5")
        target = "conv2_block1_1_conv" 
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/gmp/avg_tmr.h5")
    elif args.model == 'gmp':
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/gmp/gmp_wl_24.h5")
        target = "conv2_block1_1_conv" 
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/gmp/avg_tmr.h5")

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    if os.path.exists(base_model):
        build_tmr_model(base_model, target, output_h5)
    else:
        print(f"Error: Could not find {base_model}")