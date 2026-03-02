import os
import sys
import tensorflow as tf
import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Dynamically find the repository root and its parent directory
current_dir = os.path.abspath(os.getcwd())

if os.path.basename(current_dir) == "human_activity_recognition":
    repo_root = current_dir
elif "human_activity_recognition" in os.listdir(current_dir):
    repo_root = os.path.join(current_dir, "human_activity_recognition")
else:
    # Fallback if running from somewhere else entirely
    repo_root = current_dir

# The repository uses absolute imports (e.g., 'human_activity_recognition.tf.src...')
# Therefore, Python needs the PARENT directory of the repo in its system path.
parent_dir = os.path.dirname(repo_root)
sys.path.append(parent_dir)

# We also add the src folder just in case there are relative imports
src_path = os.path.join(repo_root, "tf", "src")
sys.path.append(src_path)

print(f"Added to Python Path: {parent_dir}")
print(f"Added to Python Path: {src_path}")

# 2. Import using the absolute path to match your repository's structure
from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm
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
def build_adaptive_clipper_model(original_model_path, representative_data, save_path):
    print(f"\nLoading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)
    
    layer_max_dict = profile_activations(model, representative_data)
    
    print("\n--- PHASE 2: REBUILDING WITH ADAPTIVE CLIPPERS ---")
    input_layer = tf.keras.Input(shape=model.input_shape[1:])
    x = input_layer
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
            
        config = layer.get_config()
        
        # 1. Robust check for fused ReLUs (Conv2D/Dense)
        has_fused_relu = False
        if 'activation' in config:
            act = config['activation']
            if isinstance(act, str) and act.lower() == 'relu':
                has_fused_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                has_fused_relu = True
                
        # 2. Check for standalone Activation('relu') layers
        is_activation_relu = False
        if isinstance(layer, tf.keras.layers.Activation):
            act = config.get('activation')
            if isinstance(act, str) and act.lower() == 'relu':
                is_activation_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                is_activation_relu = True

        # 3. Check for standalone ReLU layers
        is_standalone_relu = isinstance(layer, tf.keras.layers.ReLU)
        
        # If it has a fused ReLU, we strip it out so we can add our custom one
        if has_fused_relu:
            if isinstance(config['activation'], dict):
                config['activation']['config'] = 'linear'
                config['activation']['name'] = 'linear'
            else:
                config['activation'] = 'linear'
                
        # If it's a standalone Activation('relu') layer, we skip adding the original
        if is_activation_relu:
            pass # We don't add the original layer to 'x', we will just add the clipper below
        else:
            # Recreate the layer and apply it to 'x'
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())
        
        # ADD THE ADAPTIVE CLIPPER
        if has_fused_relu or is_standalone_relu or is_activation_relu:
            max_val = layer_max_dict[layer.name]
            safe_max_val = max(max_val, 0.1) 
            
            # Implements HardTanH(x, max(x))
            x = tf.keras.layers.ReLU(max_value=safe_max_val, name=f"{layer.name}_clipper")(x)
            print(f"Added Adaptive Clipper to {layer.name:<15} (Ceiling: {safe_max_val:.4f})")
            
    clipper_model = tf.keras.Model(inputs=input_layer, outputs=x, name="HAR_AdaptiveClipper")
    
    print(f"\nSaving hardened model to: {save_path}")
    clipper_model.save(save_path)
    print("Done!")
    
    return clipper_model

if __name__ == "__main__":
    # Based on the JSON file, this is the actual model you are validating
    base_model = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_ign/ST_pretrainedmodel_public_dataset/WISDM/st_ign_wl_24/st_ign_wl_24.keras"
    output_h5 = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_ign/ST_pretrainedmodel_public_dataset/WISDM/st_ign_wl_24/HAR_adaptive_clipper.keras"
    
    dataset_path = "human_activity_recognition/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    class_names = ['Walking', 'Jogging', 'Stairs', 'Stationary'] 
    target_shape = (24, 3, 1) # Window Length of 24, 3 axes
    
    print(f"Loading real WISDM dataset from {dataset_path}...")
    
    # Generate the actual TensorFlow datasets using your repo's utility
    train_ds, valid_ds, test_ds = load_wisdm(
        dataset_path=dataset_path,
        class_names=class_names,
        input_shape=target_shape,
        gravity_rot_sup=True,
        normalization=True,
        val_split=0.2,
        test_split=0.2,
        seed=42,
        batch_size=200, # Load 200 samples for profiling
        to_cache=False
    )
    
    if train_ds is None:
        print("Failed to load dataset. Check the dataset path.")
        sys.exit(1)
        
    print("Extracting representative data batch from the training set...")
    # Take 1 full batch (200 real physical samples) from the training dataset
    for x_batch, y_batch in train_ds.take(1):
        representative_data = x_batch.numpy()
        
    print(f"Representative data shape extracted: {representative_data.shape}")

    if os.path.exists(base_model):
        build_adaptive_clipper_model(base_model, representative_data, output_h5)
    else:
        print(f"Error: Could not find {base_model}. Make sure the path is correct.")