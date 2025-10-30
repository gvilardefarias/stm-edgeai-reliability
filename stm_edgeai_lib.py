import os
import json

model_path = os.getcwd() + "/models/gmp_wl_24/gmp_wl_24.h5"
dataset_path = os.getcwd() + "/datasets/gmp/val_data.npy"
target = "stellar-e"

weights_c_file = "./st_ai_output/src/network_data_params.c"

generate_cmd = f"stedgeai generate --model {model_path} --target {target}"
#validade_cmd = f"stedgeai validate --model {model_path} --target {target}"
validade_cmd = f"stedgeai validate --model {model_path} --target {target} --quiet -v 0 -vi {dataset_path}"
#validade_cmd = f"stedgeai validate --model {model_path} --target {target} --quiet -v 0 -b 50"

files_to_build = ["network_data", "network_data_params"]
compile_cmd  = "make clean && make all && make install && cd ../../../"

def compile_lib(build_path = "./st_ai_ws/inspector_network/workspace/"):
    os.system(f"cd {build_path} && " + compile_cmd)

def gen_lib():
    os.system(generate_cmd)
    os.system(validade_cmd)

    for f in files_to_build:
        os.system(f"cp ./st_ai_output/src/{f}.c ./st_ai_ws/inspector_network/workspace/generated/")
        os.system(f"cp ./st_ai_output/inc/{f}.h ./st_ai_ws/inspector_network/workspace/generated/")
    
    compile_lib()

    return "./st_ai_ws"

def validade_lib(lib_path = "./st_ai_ws/inspector_network/workspace/", exec_path = "./"):
    validade_lib_cmd = validade_cmd + f" --mode target -d lib:{lib_path}"
    os.system(f"cd {exec_path} &&" + validade_lib_cmd)


def get_report(exec_path = "./st_ai_ws/"):
    with open(f"{exec_path}network_report.json", 'r') as f:
        report = json.load(f)
    
    return dict(report)

def get_x_cross_accuracy(exec_path = "./st_ai_ws/"):
    report = get_report(exec_path)

    for metric in report['val_metrics']:
        if "X-cross" in metric['desc']:
            return metric['acc']

    return None


def weights_parser(weights_file = weights_c_file):
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
                        weights.append(int(num,0))
    return weights

def w_to_string(weights):
    str_o = ""

    for w in weights:
        str_o += f"{hex(w)}U,"
    
    str_o = str_o[:-1] + "\n"

    return str_o

def weights_file_gen(weights, out_file = "network_data_params.c", weights_file = weights_c_file):
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


def get_c_model_info(c_model_info_file = "./st_ai_ws/network_c_graph.json"):
    with open(c_model_info_file, 'r') as f:
        model_info = json.load(f)
    
    return dict(model_info)

def get_layers_info(c_model_info_file = "./st_ai_ws/network_c_graph.json"):
    c_model_info = get_c_model_info(c_model_info_file)

    layers_info = c_model_info['weights']['weights_array']['buffer_offsets']

    return layers_info