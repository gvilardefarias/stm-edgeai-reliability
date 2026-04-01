import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from omegaconf import OmegaConf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def profile_activations(model, representative_data):
    print("--- PROFILING MAXIMUM ACTIVATIONS ---")
    # Create a sub-model that outputs the feature maps of every layer
    extractor = tf.keras.Model(inputs=model.inputs, 
                               outputs=[layer.output for layer in model.layers])
    
    # Run the real representative data through the network
    feature_maps = extractor.predict(representative_data, verbose=0)
    
    layer_max_dict = {}
    for layer, fmap in zip(model.layers, feature_maps):
        # Calculate the absolute maximum value produced by this layer
        max_val = float(np.max(fmap))
        layer_max_dict[layer.name] = max_val
        print(f"Layer: {layer.name:<20} | Max Activation: {max_val:.4f}")
        
    return layer_max_dict

def build_adaptive_clipper_model(original_model_path, representative_data, save_path, margin=1.1):
    print(f"\nLoading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    layer_max_dict = profile_activations(model, representative_data)
    
    print("\n--- PHASE 2: REBUILDING WITH ADAPTIVE CLIPPERS ---")
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

        # Detect all forms of ReLU
        config = layer.get_config()
        is_relu = False
        
        if isinstance(layer, tf.keras.layers.ReLU):
            is_relu = True
        elif isinstance(layer, tf.keras.layers.Activation) and layer.activation.__name__ == 'relu':
            is_relu = True
        elif hasattr(layer, 'activation') and layer.activation is not None and getattr(layer.activation, '__name__', None) == 'relu':
            is_relu = True

        has_fused_relu = False
        if 'activation' in config:
            act = config['activation']
            if isinstance(act, str) and act.lower() == 'relu':
                has_fused_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                has_fused_relu = True
                
        is_activation_layer_relu = False
        if isinstance(layer, tf.keras.layers.Activation):
            act = config.get('activation')
            if isinstance(act, str) and act.lower() == 'relu':
                is_activation_layer_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                is_activation_layer_relu = True

        is_standalone_relu_layer = isinstance(layer, tf.keras.layers.ReLU)
        
        if has_fused_relu:
            if isinstance(config['activation'], dict):
                config['activation']['config'] = 'linear'
                config['activation']['name'] = 'linear'
            else:
                config['activation'] = 'linear'
                
        # Reconstruction
        if is_activation_layer_relu:
            x_out = layer_inputs # Skip original activation layer
        else:
            l_obj = layer.__class__.from_config(config)
            x_out = l_obj(layer_inputs)
            l_obj.set_weights(layer.get_weights())
        
        # Add the Adaptive Clipper
        if has_fused_relu or is_standalone_relu_layer or is_activation_layer_relu:
            max_val = layer_max_dict[layer.name]
            safe_max_val = max(max_val * margin, 0.1) 
            
            x_out = tf.keras.layers.ReLU(max_value=safe_max_val, name=f"{layer.name}_clipper")(x_out)
            print(f"Added Adaptive Clipper to {layer.name:<15} (Ceiling: {safe_max_val:.4f})")
            
        # Map output tensors
        if isinstance(layer.output, list):
            for i, out in enumerate(layer.output):
                network_dict[out.name] = x_out[i]
        else:
            network_dict[layer.output.name] = x_out

    clipper_model = tf.keras.Model(inputs=input_layer, outputs=network_dict[model.output.name], name="HAR_AdaptiveClipper")
    
    print(f"\nSaving hardened model to: {save_path}")
    clipper_model.save(save_path)
    print("Done!")
    
    return clipper_model

if __name__ == "__main__":
    # Dynamic project root detection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    # Default modelzoo path in home directory
    default_modelzoo_path = os.path.expanduser("~/stm32ai-modelzoo-services")

    parser = argparse.ArgumentParser(description='Generate Adaptive Clipper hardened models.')
    parser.add_argument('--model', type=str, default='ign', choices=['ign', 'hand_posture', 'miniresnet'], help='Model type')
    parser.add_argument('--modelzoo-path', type=str, default=default_modelzoo_path, help='Path to stm32ai-modelzoo-services')
    args = parser.parse_args()

    modelzoo_path = args.modelzoo_path
    margin = 1.1

    if args.model == 'ign':
        print(f"--- CONFIGURING FOR IGN MODEL (ModelZoo: {modelzoo_path}) ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/ign/ign_wl_24.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/ign/adaptive_clipper.h5")
        
        sys.path.append(modelzoo_path)
        from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm
        
        dataset_path = os.path.join(modelzoo_path, "human_activity_recognition/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
        class_names = ['Walking', 'Jogging', 'Stairs', 'Stationary'] 
        target_shape = (24, 3, 1)
        
        train_ds, _, _ = load_wisdm(
            dataset_path=dataset_path,
            class_names=class_names,
            input_shape=target_shape,
            gravity_rot_sup=True,
            normalization=True,
            val_split=0.2,
            test_split=0.2,
            seed=42,
            batch_size=200,
            to_cache=False
        )
        for x_batch, _ in train_ds.take(1):
            representative_data = x_batch.numpy()

    elif args.model == 'hand_posture':
        print(f"--- CONFIGURING FOR HAND POSTURE MODEL (ModelZoo: {modelzoo_path}) ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/hand_posture/adaptive_clipper.h5")
        
        sys.path.insert(0, modelzoo_path)
        from hand_posture.tf.wrappers.datasets.st_handposture import get_ST_handposture_dataset
        
        config_path = os.path.join(modelzoo_path, "hand_posture", "user_config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.dataset.training_path = os.path.join(modelzoo_path, "hand_posture/datasets/ST_VL53L8CX_handposture_dataset")
        
        data_loaders = get_ST_handposture_dataset(cfg)
        train_ds = data_loaders['train']
        for images, labels in train_ds.take(1):
            representative_data = images.numpy()

    elif args.model == 'miniresnet':
        print("--- CONFIGURING FOR MINIRESNET MODEL ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/miniresnet/adaptive_clipper.h5")
        
        # MiniResNet uses a straight .npy dataset
        dataset_path = os.path.join(PRJ_ROOT, "sw/datasets/miniresnet/miniresnet_dataset.npy")
        representative_data = np.load(dataset_path)[:100] # Take first 100 samples for profiling
        print(f"Loaded {representative_data.shape[0]} samples for profiling.")

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    if os.path.exists(base_model):
        build_adaptive_clipper_model(base_model, representative_data, output_h5, margin=margin)
    else:
        print(f"Error: Could not find {base_model}")