import os
import json

model_path = os.getcwd() + "/hardening/hardened_models/gmp/HAR_tmr_voter.h5"
dataset_path = os.getcwd() + "/datasets/gmp/val_data.npy"
dataset_label = os.getcwd() + "/datasets/gmp/val_labels.npy"
#model_path = os.getcwd() + "/models/miniresnet/miniresnet_1stacks_64x50_tl.h5"
#dataset_path = os.getcwd() + "/datasets/miniresnet/miniresnet_dataset.npy"
target = "stellar-e"
workspace_path = "/st_ai_ws/inspector_network/workspace/"

weights_c_file = "./st_ai_output/src/network_data_params.c"

config_path = "/home/apo/stm-edgeai-reliability/hardening/config/voter_tmr.json"

generate_cmd = f"stedgeai generate --model {model_path} --target {target} --custom {config_path}"


# Assuming your loop variable for the current fault path is something like 'fault_path'
# Example: fault_path = "/home/apo/stm-edgeai-reliability/fault_campaign/sta0_w0_b63"

library_path = f"st_ai_ws/inspector_network/workspace/lib/libai_network.so"

validade_cmd = (
    f"stedgeai validate --model {model_path} "
    f"--custom {config_path} "
    f"--target {target} "
    f"--quiet -v 0 -vi {dataset_path} --valoutput{dataset_label} "
)

files_to_build = ["network_data", "network_data_params"]
compile_cmd = "make clean && make all && make install && cd ../../../"

def compile_lib(build_path="./st_ai_ws/inspector_network/workspace/"):
    print("compile", build_path)
    os.system(f"cd {build_path} && " + compile_cmd)

def gen_lib():
    # 1. Generate the C code and weights from the .h5 model
    os.system(generate_cmd)

    # 2. Copy the newly generated weight data into the build workspace
    for f in files_to_build:
        os.system(f"cp ./st_ai_output/src/{f}.c ./st_ai_ws/inspector_network/workspace/generated/")
        os.system(f"cp ./st_ai_output/inc/{f}.h ./st_ai_ws/inspector_network/workspace/generated/")
    
    # 3. Compile the shared library (.so) so the weights are embedded
    compile_lib()

    # 4. Validate with --mode target so stedgeai writes network_report.json
    # into ./st_ai_ws/ (the exec_path). The bare validade_cmd (without --mode target)
    # does NOT write network_report.json there, which caused the FileNotFoundError
    # when reading the golden report later.
    validade_lib(
        lib_path="./st_ai_ws/inspector_network/workspace/",
        exec_path="./st_ai_ws/"
    )

    return "./st_ai_ws"

def validade_lib(lib_path="./st_ai_ws/inspector_network/workspace/", exec_path="./"):
    # BUG1 FIX: stedgeai requires --mode target -d lib: to point to the actual
    # .so file (with weights embedded), not to the workspace directory.
    # The Makefile installs the library to lib/libai_network.so inside lib_path.
    # stedgeai also requires an ABSOLUTE path — relative paths cause E200.
    so_path = os.path.abspath(os.path.join(lib_path, "lib/libai_network.so"))
    validade_lib_cmd = validade_cmd + f"--mode target -d lib:{so_path}"
    print("validade", validade_lib_cmd)
    os.system(f"cd {exec_path} && " + validade_lib_cmd)


def get_report(exec_path="./st_ai_ws/"):
    with open(f"{exec_path}network_report.json", 'r') as f:
        report = json.load(f)
    
    return dict(report)

def get_x_cross_accuracy(exec_path="./st_ai_ws/"):
    report = get_report(exec_path)

    for metric in report['val_metrics']:
        if "X-cross" in metric['desc']:
            return metric['acc'], metric

    return None, None


def weights_parser(weights_file=weights_c_file):
    weights = []
    with open(weights_file, 'r') as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if "s_network_weights_array_u64" in line:
                start = True
                continue
            if start:
                if "};" in line:
                    break

                line = line.replace(" ", "")
                line = line.replace("U", "")
                line = line.replace("\n", "")
                nums = line.split(",")
                for num in nums:
                    if num != "":
                        weights.append(int(num, 0))
    return weights

def w_to_string(weights):
    str_o = ""

    for w in weights:
        str_o += f"{hex(w)}U,"
    
    str_o = str_o[:-1] + "\n"

    return str_o

def weights_file_gen(weights, out_file="network_data_params.c", weights_file=weights_c_file):
    str_file = ""

    with open(weights_file, 'r') as f:
        lines = f.readlines()
        start = False
        stop = False
        for line in lines:
            if "s_network_weights_array_u64[" in line and "{" in line:
                start = True
                str_file += line
            elif start and not stop:
                if "};" in line:
                    str_file += w_to_string(weights)
                    str_file += line
                    stop = True
            elif start and stop:
                str_file += line
            else:
                str_file += line

    with open(out_file, 'w') as o:
        o.write(str_file)


def get_c_model_info(c_model_info_file="./st_ai_ws/network_c_graph.json"):
    with open(c_model_info_file, 'r') as f:
        model_info = json.load(f)
    
    return dict(model_info)

def set_layers_info(layers_info, c_model_info_file="./st_ai_ws/network_c_graph.json"):
    c_model_info = get_c_model_info(c_model_info_file)

    c_model_info['weights']['weights_array']['buffer_offsets'] = layers_info

    with open(c_model_info_file, 'w') as f:
        json.dump(c_model_info, f, indent=4)

def get_layers_info(c_model_info_file="./st_ai_ws/network_c_graph.json"):
    c_model_info = get_c_model_info(c_model_info_file)

    layers_info = c_model_info['weights']['weights_array']['buffer_offsets']

    return layers_info