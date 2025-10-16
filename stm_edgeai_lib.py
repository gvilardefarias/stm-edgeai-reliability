import os

model_path = os.getcwd() + "/models/gmp_wl_24/gmp_wl_24.h5"
target = "stellar-e"

weights_c_file = "./st_ai_output/src/network_data_params.c"

generate_cmd = f"stedgeai generate --model {model_path} --target {target}"
validade_cmd = f"stedgeai validate --model {model_path} --target {target}"

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

def validade_lib(lib_path = "./st_ai_ws/inspector_network/workspace/"):
    validade_lib_cmd = validade_cmd + f" --mode target -d lib:{lib_path}"
    os.system(validade_lib_cmd)

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