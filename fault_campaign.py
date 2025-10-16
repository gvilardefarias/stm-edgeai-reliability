from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import stm_edgeai_lib
from contextlib import chdir

n_th = 32

def inject_sta_fault(data, f_type, f_idx, f_bit):
    if f_type == "sta0":
        data[f_idx] &= ~(1 << f_bit)
    elif f_type == "sta1":
        data[f_idx] |= (1 << f_bit)
    
    return data


def simulate_fault(weights, f_type, f_idx, f_bit):
    lib_path = "./inspector_network/workspace/"
    f_id = f"{f_type}_w{f_idx}_b{f_bit}"

    os.makedirs(f"./fault_campaign/{f_id}", exist_ok=True)
    os.system(f"cp -r ./fault_campaign/golden/* ./fault_campaign/{f_id}/")
    
    with chdir(f"./fault_campaign/{f_id}"):
        faulty_weights = inject_sta_fault(weights.copy(), f_type, f_idx, f_bit)
        stm_edgeai_lib.weights_file_gen(faulty_weights, weights_file="../../st_ai_output/src/network_data_params.c")
        os.system(f"rm {lib_path}generated/network_data_params.c")
        os.system(f"mv ./network_data_params.c {lib_path}generated/")
        stm_edgeai_lib.compile_lib(lib_path)
        stm_edgeai_lib.validade_lib(lib_path)

        #stm_edgeai_lib.get_accuracy(f"")


def sta_fault_campaign(f_bit_range = 1):
    golden_lib = stm_edgeai_lib.gen_lib()

    os.makedirs("./fault_campaign", exist_ok=True)
    os.makedirs("./fault_campaign/golden", exist_ok=True)

    os.system("rm -rf ./fault_campaign/*")
    os.system(f"cp -r {golden_lib} ./fault_campaign/golden/")

    with ProcessPoolExecutor(max_workers=n_th) as executor:
        futures = []
        weights = stm_edgeai_lib.weights_parser()

        for f_idx in range(len(weights)):
            for f_bit in range(64-1, 64-1-f_bit_range, -1):
                for f_type in ["sta0", "sta1"]:
                    faulty_weights = inject_sta_fault(weights.copy(), f_type, f_idx, f_bit)
                    futures.append(executor.submit(simulate_fault, faulty_weights, f_type, f_idx, f_bit))

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

sta_fault_campaign()