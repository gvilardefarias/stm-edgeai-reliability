import os
import re
import struct
import argparse
import numpy as np
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def float32_array_to_bytes(arr):
    return arr.astype(np.float32).flatten().tobytes()

def get_offset_from_network_c(network_c, array_name):
    # e.g., array_name = "conv2d_tmr1_weights" or "conv2d_1_tmr1_weights"
    pattern = rf'{array_name}_array\.data_start = AI_PTR\(g_network_weights_map\[0\] \+ (\d+)\);'
    with open(network_c, 'r') as f:
        match = re.search(pattern, f.read())
    if match:
        return int(match.group(1))
    return None

def patch_c_weights(base_model_path, target_layer_name, st_ai_output_dir):
    c_file = os.path.join(st_ai_output_dir, 'src', 'network_data_params.c')
    net_c_file = os.path.join(st_ai_output_dir, 'src', 'network.c')
    
    if not os.path.exists(c_file) or not os.path.exists(net_c_file):
        print(f"Error: Could not find STEdgeAI generated C code at {st_ai_output_dir}")
        return
        
    print(f"Reading {c_file}...")
    with open(c_file, 'r') as f:
        content = f.read()
    
    # Extract the u64 array and its contents
    # We use a regex that matches the declaration and grabs the inside
    match = re.search(r'(const ai_u64 s_network_weights_array_u64\[\d+\] = \{)(.*?)(^\};)', content, re.DOTALL | re.MULTILINE)
    if not match:
        print("Error: Could not find weights array in C file")
        return
        
    prefix = match.group(1)
    array_content = match.group(2)
    suffix = match.group(3)
    
    # Find all hex tokens
    hex_strs = re.findall(r'0x([0-9a-fA-F]+)U', array_content)
    
    # Convert entire array to bytearray
    c_bytes = bytearray()
    for h in hex_strs:
        h = h.zfill(16)
        val = int(h, 16)
        c_bytes.extend(struct.pack('<Q', val))
    
    # 2. Get pristine weights from base Keras model
    print(f"Loading pristine weights from {base_model_path}...")
    model = tf.keras.models.load_model(base_model_path)
    layer = model.get_layer(target_layer_name)
    weights_biases = layer.get_weights()
    if len(weights_biases) == 0:
        print("Layer has no weights.")
        return
        
    weights = weights_biases[0]
    bias = weights_biases[1] if len(weights_biases) > 1 else None
    print("Converting pristine weights to bytes...")
    
    # Transpose Keras weights to ST Edge AI format: (out_channels, filter_h, filter_w, in_channels)
    # This works for Conv2D. For Dense, it's (out_channels, in_channels).
    if len(weights.shape) == 4:
        keras_w_transposed = np.transpose(weights, (3, 0, 1, 2)).flatten()
    elif len(weights.shape) == 2: # Dense
        keras_w_transposed = np.transpose(weights, (1, 0)).flatten()
    else:
        keras_w_transposed = weights.flatten()
        
    perfect_w_bytes = float32_array_to_bytes(keras_w_transposed)
    w_size = len(perfect_w_bytes)
    
    perfect_b_bytes = b""
    b_size = 0
    if bias is not None:
        perfect_b_bytes = float32_array_to_bytes(bias.flatten())
        b_size = len(perfect_b_bytes)
    
    # 3. Patch the C bytearray at the correct offsets discovered from network.c
    print("Patching the epsilon differences in byte array for all 3 TMR branches...")
    for tmr_idx in [1, 2, 3]:
        # Formulate the layer array names. E.g., target_layer_name = "conv2d" -> "conv2d_tmr1_weights"
        w_array_name = f"{target_layer_name}_tmr{tmr_idx}_weights"
        b_array_name = f"{target_layer_name}_tmr{tmr_idx}_bias"
        
        w_offset = get_offset_from_network_c(net_c_file, w_array_name)
        if w_offset is None:
            # Fallback if the naming has the base layer name wrapped inside tmr. Like conv2d_1 -> conv2d_1_tmr1
            w_array_name = f"{target_layer_name}_tmr{tmr_idx}_weights"
            w_offset = get_offset_from_network_c(net_c_file, w_array_name)
            
        if w_offset is None:
            # Maybe this is a BIAS TMR, the weights are not triplicated, only bias is. 
            pass # Valid
        else:
            c_bytes[w_offset : w_offset + w_size] = perfect_w_bytes
            print(f"Patched {w_array_name} at offset {w_offset}")
        
        if bias is not None:
            b_offset = get_offset_from_network_c(net_c_file, b_array_name)
            if b_offset is None:
                # Custom layer bias TMR fallback: e.g. "conv2d_tmr_b_param_0"
                b_array_name = f"{target_layer_name}_tmr_b_param_{tmr_idx - 1}"
                b_offset = get_offset_from_network_c(net_c_file, b_array_name)
                
            if b_offset is not None:
                c_bytes[b_offset : b_offset + b_size] = perfect_b_bytes
                print(f"Patched {b_array_name} at offset {b_offset}")
    
    # 4. Reconstruct hex strings
    print("Rebuilding C array...")
    new_hex_strings = []
    for i in range(0, len(c_bytes), 8):
        chunk = c_bytes[i:i+8]
        if len(chunk) < 8:
            chunk += b'\x00' * (8 - len(chunk))
        val = struct.unpack('<Q', chunk)[0]
        hex_str = f"  0x{val:016x}U,"
        new_hex_strings.append(hex_str)
        
    formatted_array_content = "\n"
    for i in range(0, len(new_hex_strings), 4):
        formatted_array_content += " ".join(new_hex_strings[i:i+4]) + "\n"
        
    # 5. Overwrite the file
    new_content = content[:match.start()] + prefix + formatted_array_content + suffix + content[match.end():]
    
    print(f"Writing updated pristine values back to {c_file}...")
    with open(c_file, 'w') as f:
        f.write(new_content)
        
    print("Patch completed successfully! STEdgeAI deduplication is bypassed and true hardware redundancy logic has identical weights restored!")

if __name__ == '__main__':
    import argparse
    from argparse import RawTextHelpFormatter
    
    desc = """
Patch ST Edge AI weights with mathematically pristine bits to bypass TMR deduplication.

EXAMPLES:
  1. Voter TMR (targets 'conv2d'):
     python patch_c_weights.py \\
       --base-model sw/hardening/base_models/gmp/gmp_wl_24.h5 \\
       --target-layer conv2d \\
       --st-ai-output st_ai_output/

  2. Average/Median TMR (targets 'conv2d_1'):
     python patch_c_weights.py \\
       --base-model sw/hardening/base_models/gmp/gmp_wl_24.h5 \\
       --target-layer conv2d_1 \\
       --st-ai-output st_ai_output/

  3. Bias TMR (targets 'conv2d' bias only):
     python patch_c_weights.py \\
       --base-model sw/hardening/base_models/gmp/gmp_wl_24.h5 \\
       --target-layer conv2d \\
       --st-ai-output st_ai_output/
"""
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model", default="hand_posture", choices=["ign", "hand_posture", "miniresnet"], help="Model type")
    parser.add_argument("--st-ai-output", default="/home/apo/stm-edgeai-reliability/st_ai_output", help="Directory containing generated C files")
    
    args = parser.parse_args()
    
    if args.model == 'ign':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/ign/ign_wl_24.h5"
        target = "conv2d"
    elif args.model == 'hand_posture':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5"
        target = "conv2d"
    elif args.model == 'miniresnet':
        base_model = "/home/apo/stm-edgeai-reliability/sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5"
        target = "conv2_block1_1_conv"

    patch_c_weights(base_model, target, args.st_ai_output)
