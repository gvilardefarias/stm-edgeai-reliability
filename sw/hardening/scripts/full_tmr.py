import sys
import os
import tensorflow as tf
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_layers')))
from majority_voter_layer import MajorityVoterLayer


def clone_model_branch(model, branch_input, branch_idx):
    """
    Rebuild the full model graph on top of branch_input.
    branch_idx 0 => tmr1, epsilon=0  (golden reference for the patch script)
    branch_idx 1 => tmr2, epsilon=1e-6
    branch_idx 2 => tmr3, epsilon=2e-6
    Returns the output tensor of the cloned branch.
    """
    epsilon = branch_idx * 1e-6

    network_dict = {model.input.name: branch_input}

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
        config['name'] = f"{layer.name}_tmr{branch_idx + 1}"  # tmr1 / tmr2 / tmr3
        cloned_layer = layer.__class__.from_config(config)

        x_out = cloned_layer(layer_inputs)

        original_weights = layer.get_weights()
        if original_weights:
            modified_weights = []
            for w in original_weights:
                if np.issubdtype(w.dtype, np.floating):
                    modified_weights.append((w.copy() + epsilon).astype(np.float32))
                else:
                    modified_weights.append(w.copy())
            cloned_layer.set_weights(modified_weights)

        if isinstance(layer.output, list):
            for i, out in enumerate(layer.output):
                network_dict[out.name] = x_out[i]
        else:
            network_dict[layer.output.name] = x_out

    return network_dict[model.output.name]


def build_full_tmr_model(original_model_path, save_path):
    print(f"Loading original model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path)

    shared_input = tf.keras.Input(shape=model.input_shape[1:], name="input_1")

    branch_outputs = []
    for i in range(3):
        print(f"--- BUILDING BRANCH {i + 1}/3 (tmr{i + 1}, epsilon={i * 1e-6}) ---")
        branch_out = clone_model_branch(model, shared_input, branch_idx=i)
        branch_outputs.append(branch_out)

    print("--- ADDING MAJORITY VOTER AT OUTPUT ---")
    output = MajorityVoterLayer(name="output_voter")(branch_outputs)

    tmr_model = tf.keras.Model(
        inputs=shared_input,
        outputs=output,
        name=f"{model.name}_full_tmr"
    )

    tmr_model.summary()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"\nSaving full TMR model to: {save_path}")
    tmr_model.save(save_path, save_format='h5')
    print("Done!")
    return tmr_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Full TMR hardened model.')
    parser.add_argument(
        '--model', type=str, default='hand_posture',
        choices=['gmp', 'ign', 'hand_posture', 'miniresnet'],
        help='Model type'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    model_configs = {
        'ign': {
            'base': "sw/hardening/base_models/ign/ign_wl_24.h5",
            'out':  "sw/hardening/hardened_models/ign/full_tmr.h5",
        },
        'gmp': {
            'base': "sw/hardening/base_models/gmp/gmp_wl_24.h5",
            'out':  "sw/hardening/hardened_models/gmp/full_tmr.h5",
        },
        'hand_posture': {
            'base': "sw/hardening/base_models/hand_posture/CNN2D_ST_HandPosture_8classes.h5",
            'out':  "sw/hardening/hardened_models/hand_posture/full_tmr.h5",
        },
        'miniresnet': {
            'base': "sw/hardening/base_models/miniresnet/miniresnet_1stacks_64x50_tl.h5",
            'out':  "sw/hardening/hardened_models/miniresnet/full_tmr.h5",
        },
    }

    cfg = model_configs[args.model]
    base_model_path = os.path.join(PRJ_ROOT, cfg['base'])
    output_h5 = os.path.join(PRJ_ROOT, cfg['out'])

    if not os.path.exists(base_model_path):
        print(f"Error: Could not find {base_model_path}")
        sys.exit(1)

    build_full_tmr_model(base_model_path, output_h5)