import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import stm_edgeai_lib as stm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch weights with epsilon to bypass deduplication')
    parser.add_argument('--weights_file', type=str, default=stm.weights_c_file, help='Path to the original weights C file')
    parser.add_argument('--network_file', type=str, default=stm.weights_c_file, help='Path to the json file with the network information')
    args = parser.parse_args()

    weights = stm.weights_parser(args.weights_file)

    layer_info = stm.get_layers_info(args.network_file)

    golden_weights = []
    golden_bias = []
    for layer in layer_info:
        if "tmr1" in layer['buffer_name']:
            #Start and end of the weights in the weights array
            start_weight = layer['offset'] // 8 
            end_weight = (layer['offset'] + layer['size']) // 8
            if "bias" in layer['buffer_name']:
                golden_bias.extend(weights[start_weight:end_weight])
            else:
                golden_weights.extend(weights[start_weight:end_weight])

    for layer in layer_info:
        if ("tmr2" in layer['buffer_name'] or "tmr3" in layer['buffer_name']):
            start_weight = layer['offset'] // 8 
            end_weight = (layer['offset'] + layer['size']) // 8
            if "bias" in layer['buffer_name']:
                weights[start_weight:end_weight] = golden_bias
            else:
                weights[start_weight:end_weight] = golden_weights

    stm.weights_file_gen(weights, weights_file=args.weights_file)