import os
import sys
import tensorflow as tf
import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Path setup depends on argparse which is now in __main__
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

def build_adaptive_clipper_model(original_model_path, representative_data, margin=1.1):
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
        
        # Detect all forms of ReLU
        is_relu = False
        if isinstance(layer, tf.keras.layers.ReLU):
            is_relu = True
        elif isinstance(layer, tf.keras.layers.Activation) and layer.activation.__name__ == 'relu':
            is_relu = True
        elif hasattr(layer, 'activation') and layer.activation is not None and getattr(layer.activation, '__name__', None) == 'relu':
            is_relu = True

        # If it's a fused ReLU, strip it from the original layer config
        has_fused_relu = False
        if 'activation' in config:
            act = config['activation']
            if isinstance(act, str) and act.lower() == 'relu':
                has_fused_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                has_fused_relu = True
        is_activation_relu = False
        if isinstance(layer, tf.keras.layers.Activation):
            act = config.get('activation')
            if isinstance(act, str) and act.lower() == 'relu':
                is_activation_relu = True
            elif isinstance(act, dict) and (act.get('config') == 'relu' or act.get('name') == 'relu'):
                is_activation_relu = True

        # 3. Check for standalone ReLU layers
        is_standalone_relu = isinstance(layer, tf.keras.layers.ReLU)

        # Strip fused ReLU so we can replace with our clipper
        if has_fused_relu:
            if isinstance(config['activation'], dict):
                config['activation']['config'] = 'linear'
                config['activation']['name'] = 'linear'
            else:
                config['activation'] = 'linear'

        # Skip standalone Activation('relu') — we replace it with the clipper below
        if is_activation_relu:
            pass  # Don't add original layer; clipper replaces it
        else:
            new_layer = layer.__class__.from_config(config)
            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())

        if has_fused_relu or is_standalone_relu or is_activation_relu:
            max_val = layer_max_dict[layer.name]
            safe_max_val = max(max_val, 0.1)
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
    import argparse
    
    # Dynamic project root detection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    # Default modelzoo path
    default_modelzoo_path = os.path.expanduser("~/stm32ai-modelzoo-services")
    
    parser = argparse.ArgumentParser(description='Finetune Adaptive Clipper hardened models.')
    parser.add_argument('--modelzoo-path', type=str, default=default_modelzoo_path, help='Path to stm32ai-modelzoo-services')
    args = parser.parse_args()
    
    modelzoo_path = args.modelzoo_path
    
    if modelzoo_path not in sys.path:
        sys.path.append(modelzoo_path)
    
    # Now import load_wisdm after adding to path
    from human_activity_recognition.tf.src.datasets.wisdm import load_wisdm

    base_model = os.path.join(PRJ_ROOT, "sw/hardening/base_models/ign/ign_wl_24.h5")
    output_h5 = os.path.join(PRJ_ROOT, "sw/hardening/hardened_models/ign/adaptive_clipper_finetuned.h5")
    
    dataset_path = os.path.join(modelzoo_path, "human_activity_recognition/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
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