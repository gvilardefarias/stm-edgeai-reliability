import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from omegaconf import OmegaConf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def profile_activations(model, representative_data):
    print("--- PROFILING MAXIMUM ACTIVATIONS ---")
    extractor = tf.keras.Model(inputs=model.inputs, 
                               outputs=[layer.output for layer in model.layers])
    
    feature_maps = extractor.predict(representative_data, verbose=0)
    
    layer_max_dict = {}
    for layer, fmap in zip(model.layers, feature_maps):
        max_val = float(np.max(fmap))
        layer_max_dict[layer.name] = max_val
        print(f"Layer: {layer.name:<20} | Max Activation: {max_val:.4f}")
        
    return layer_max_dict

def build_adaptive_clipper_model(original_model_path, representative_data, save_path, margin=1.1, target_layers=None):
    print(f"\nLoading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path, compile=False)

    if target_layers:
        model_layer_names = {layer.name for layer in model.layers}
        invalid = [n for n in target_layers if n not in model_layer_names]
        if invalid:
            raise ValueError(f"Target layer(s) not found in model: {invalid}\n"
                             f"Available layers: {sorted(model_layer_names)}")
        print(f"\nTarget layers (clipper will ONLY be applied to these): {target_layers}")
    else:
        print("\nNo target layers specified — clipper will be applied to ALL ReLU layers.")

    layer_max_dict = profile_activations(model, representative_data)
    
    print("\n--- PHASE 2: REBUILDING WITH ADAPTIVE CLIPPERS ---")
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    network_dict = {model.input.name: input_layer}
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
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

        config = layer.get_config()

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

        is_relu_layer = has_fused_relu or is_standalone_relu_layer or is_activation_layer_relu
        should_clip = is_relu_layer and (target_layers is None or layer.name in target_layers)

        if has_fused_relu and should_clip:
            if isinstance(config['activation'], dict):
                config['activation']['config'] = 'linear'
                config['activation']['name'] = 'linear'
            else:
                config['activation'] = 'linear'

        if is_activation_layer_relu and should_clip:
            x_out = layer_inputs
        else:
            l_obj = layer.__class__.from_config(config)
            x_out = l_obj(layer_inputs)
            l_obj.set_weights(layer.get_weights())

        if should_clip:
            max_val = layer_max_dict[layer.name]
            safe_max_val = max(max_val * margin, 0.1)
            x_out = tf.keras.layers.ReLU(max_value=safe_max_val, name=f"{layer.name}_clipper")(x_out)
            print(f"  [CLIPPED] {layer.name:<25} → ceiling: {safe_max_val:.4f}")
        elif is_relu_layer:
            print(f"  [SKIPPED] {layer.name:<25} (ReLU present but not in target list)")

        if isinstance(layer.output, list):
            for i, out in enumerate(layer.output):
                network_dict[out.name] = x_out[i]
        else:
            network_dict[layer.output.name] = x_out

    clipper_model = tf.keras.Model(
        inputs=input_layer,
        outputs=network_dict[model.output.name],
        name="HAR_AdaptiveClipper"
    )
    
    print(f"\nSaving hardened model to: {save_path}")
    clipper_model.save(save_path)
    print("Done!")
    
    return clipper_model


def collect_dataset(ds):
    """Iterate over an entire tf.data.Dataset and concatenate all batches."""
    all_batches = []
    for x_batch, _ in ds:
        all_batches.append(x_batch.numpy())
    data = np.concatenate(all_batches, axis=0)
    print(f"Collected {data.shape[0]} samples for profiling.")
    return data


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    default_modelzoo_path = os.path.expanduser("~/stm32ai-modelzoo-services")

    parser = argparse.ArgumentParser(description='Generate Adaptive Clipper hardened models.')
    parser.add_argument('--model', type=str, default='ign', choices=['ign', 'hand_posture', 'miniresnet'],
                        help='Model type')
    parser.add_argument('--modelzoo-path', type=str, default=default_modelzoo_path,
                        help='Path to stm32ai-modelzoo-services')
    parser.add_argument('--target-layers', type=str, nargs='+', default=None,
                        metavar='LAYER_NAME',
                        help='Names of specific layers to apply the clipper to. '
                             'If omitted, clips ALL ReLU layers.')
    parser.add_argument('--data-split', type=str, default='val', choices=['train', 'val'],
                        help='Which data split to use for profiling activations. '
                             'Default: val (recommended). Use train to match original behaviour.')
    parser.add_argument('--val-split', type=float, default=0.2,
                        metavar='FLOAT',
                        help='Fraction of data to use as validation set for MiniResNet '
                             '(last N%% of samples). Only used when --data-split=val. Default: 0.2')
    args = parser.parse_args()

    modelzoo_path = args.modelzoo_path
    margin = 1.1
    target_layers = args.target_layers

    print(f"Profiling activations using: {args.data_split.upper()} split")

    if args.model == 'ign':
        print(f"--- CONFIGURING FOR IGN MODEL (ModelZoo: {modelzoo_path}) ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/ign/ign_wl_24.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/ign/adaptive_clipper.h5")
        
        sys.path.append(modelzoo_path)
        from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm
        
        dataset_path = os.path.join(modelzoo_path, "human_activity_recognition/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
        class_names = ['Walking', 'Jogging', 'Stairs', 'Stationary']
        target_shape = (24, 3, 1)
        
        train_ds, val_ds, _ = load_wisdm(
            dataset_path=dataset_path, class_names=class_names, input_shape=target_shape,
            gravity_rot_sup=True, normalization=True, val_split=0.2, test_split=0.2,
            seed=42, batch_size=200, to_cache=False
        )
        representative_data = collect_dataset(train_ds if args.data_split == 'train' else val_ds)

    elif args.model == 'hand_posture':
        print(f"--- CONFIGURING FOR HAND POSTURE MODEL (ModelZoo: {modelzoo_path}) ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/hand_posture/adaptive_clipper.h5")
        
        sys.path.insert(0, modelzoo_path)
        from hand_posture.tf.wrappers.datasets.st_handposture import get_ST_handposture_dataset
        
        config_path = os.path.join(modelzoo_path, "hand_posture", "user_config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.dataset.training_path = os.path.join(modelzoo_path, "hand_posture/datasets/ST_VL53L8CX_handposture_dataset")
        if cfg.dataset.validation_split is None:
            cfg.dataset.validation_split = 0.2
        
        data_loaders = get_ST_handposture_dataset(cfg)
        split_key = 'train' if args.data_split == 'train' else 'valid'
        representative_data = collect_dataset(data_loaders[split_key])

    elif args.model == 'miniresnet':
        print("--- CONFIGURING FOR MINIRESNET MODEL ---")
        base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5")
        output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/miniresnet/adaptive_clipper.h5")
        
        dataset_path = os.path.join(PRJ_ROOT, "sw/datasets/miniresnet/miniresnet_dataset.npy")
        full_data = np.load(dataset_path)
        
        if args.data_split == 'train':
            split_idx = int(len(full_data) * (1.0 - args.val_split))
            representative_data = full_data[:split_idx]
            print(f"Using first {1 - args.val_split:.0%} as train split "
                  f"({representative_data.shape[0]}/{len(full_data)} samples) for profiling.")
        else:
            split_idx = int(len(full_data) * (1.0 - args.val_split))
            representative_data = full_data[split_idx:]
            print(f"Using last {args.val_split:.0%} as validation split "
                  f"({representative_data.shape[0]}/{len(full_data)} samples) for profiling.")

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    if os.path.exists(base_model):
        build_adaptive_clipper_model(base_model, representative_data, output_h5,
                                     margin=margin, target_layers=target_layers)
    else:
        print(f"Error: Could not find {base_model}")