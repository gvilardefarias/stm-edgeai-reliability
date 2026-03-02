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
    repo_root = current_dir

parent_dir = os.path.dirname(repo_root)
sys.path.append(parent_dir)

src_path = os.path.join(repo_root, "tf", "src")
sys.path.append(src_path)

from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm

def profile_activations(model, representative_data):
    print("\n--- PHASE 1: PROFILING MAXIMUM ACTIVATIONS ---")
    extractor = tf.keras.Model(inputs=model.inputs, 
                               outputs=[layer.output for layer in model.layers])
    
    feature_maps = extractor.predict(representative_data, verbose=0)
    
    layer_max_dict = {}
    for layer, fmap in zip(model.layers, feature_maps):
        max_val = float(np.max(fmap))
        layer_max_dict[layer.name] = max_val
        print(f"Layer: {layer.name:<20} | Max Activation: {max_val:.4f}")
        
    return layer_max_dict

def build_adaptive_clipper_model(original_model_path, representative_data):
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
        has_fused_relu = 'activation' in config and config['activation'] == 'relu'
        
        if has_fused_relu:
            config['activation'] = 'linear'
            
        new_layer = layer.__class__.from_config(config)
        x = new_layer(x)
        new_layer.set_weights(layer.get_weights())
        
        is_standalone_relu = isinstance(layer, tf.keras.layers.ReLU) and layer.max_value is None
        
        if has_fused_relu or is_standalone_relu:
            max_val = layer_max_dict[layer.name]
            safe_max_val = max(max_val, 0.1) 
            # Implements HardTanH(x, max(x)) from the paper
            x = tf.keras.layers.ReLU(max_value=safe_max_val, name=f"{layer.name}_clipper")(x)
            print(f"Added Adaptive Clipper to {layer.name:<15} (Ceiling: {safe_max_val:.4f})")
            
    clipper_model = tf.keras.Model(inputs=input_layer, outputs=x, name="HAR_AdaptiveClipper")
    return clipper_model

def finetune_clipper_model(model, train_ds, valid_ds, save_path, epochs=10, learning_rate=1e-4):
    print("\n--- PHASE 3: FINE-TUNING THE HARDENED MODEL ---")
    
    # We use a low learning rate to gently adjust weights to the new clipping bounds
    # without destroying the pre-trained feature extraction capabilities.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile the model (adjust loss to sparse_categorical_crossentropy if your labels aren't one-hot encoded)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Callbacks to save the best fine-tuned model and stop early if it plateaus
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    print(f"Starting fine-tuning for up to {epochs} epochs...")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print(f"\nFine-tuned hardened model saved to: {save_path}")
    return model

if __name__ == "__main__":
    base_model = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_gmp/ST_pretrainedmodel_public_dataset/WISDM/st_gmp_wl_24/st_gmp_wl_24.keras"
    output_h5 = "/home/apo/stm32ai-modelzoo/human_activity_recognition/st_gmp/ST_pretrainedmodel_public_dataset/WISDM/st_gmp_wl_24/HAR_adaptive_clipper_finetuned.h5"
    
    dataset_path = "human_activity_recognition/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    class_names = ['Walking', 'Jogging', 'Stairs', 'Stationary'] 
    target_shape = (24, 3, 1) 
    
    print(f"Loading real WISDM dataset from {dataset_path}...")
    
    train_ds, valid_ds, test_ds = load_wisdm(
        dataset_path=dataset_path,
        class_names=class_names,
        input_shape=target_shape,
        gravity_rot_sup=True,
        normalization=True,
        val_split=0.2,
        test_split=0.2,
        seed=42,
        batch_size=64, # Reduced from 200 for better fine-tuning gradient updates
        to_cache=False
    )
    
    if train_ds is None:
        print("Failed to load dataset. Check the dataset path.")
        sys.exit(1)
        
    print("Extracting representative data batch for profiling...")
    for x_batch, y_batch in train_ds.take(1):
        representative_data = x_batch.numpy()

    if os.path.exists(base_model):
        # Phase 1 & 2: Profile and Build
        clipper_model = build_adaptive_clipper_model(base_model, representative_data)
        
        # Phase 3: Fine-Tune
        finetuned_model = finetune_clipper_model(
            model=clipper_model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            save_path=output_h5,
            epochs=10,           # Adjust based on how long it takes to converge
            learning_rate=1e-4   # Small LR to preserve pre-trained weights
        )
    else:
        print(f"Error: Could not find {base_model}. Make sure the path is correct.")