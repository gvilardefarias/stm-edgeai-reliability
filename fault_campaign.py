from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import stm_edgeai_lib
from contextlib import contextmanager

n_th = 32

def inject_sta_fault(data, f_type, f_idx, f_bit):
    if f_type == "sta0":
        data[f_idx] &= ~(1 << f_bit)
    elif f_type == "sta1":
        data[f_idx] |= (1 << f_bit)
    
    return data


def simulate_fault(weights, f_type, f_idx, f_bit):
    f_id = f"{f_type}_w{f_idx}_b{f_bit}"

    work_path = os.getcwd() + f"/fault_campaign/{f_id}/"
    lib_path = f"{work_path}inspector_network/workspace/"

    os.makedirs(f"./fault_campaign/{f_id}", exist_ok=True)
    os.system(f"cp -r ./fault_campaign/golden/* ./fault_campaign/{f_id}/")
    
    faulty_weights = inject_sta_fault(weights.copy(), f_type, f_idx, f_bit)
    os.system(f"rm {lib_path}generated/network_data_params.c")
    stm_edgeai_lib.weights_file_gen(faulty_weights, out_file=f"{lib_path}generated/network_data_params.c", weights_file=f"{work_path}../../st_ai_output/src/network_data_params.c")
    stm_edgeai_lib.compile_lib(lib_path)
    stm_edgeai_lib.validade_lib(lib_path, exec_path=work_path)

    return (stm_edgeai_lib.get_x_cross_accuracy(f"{work_path}st_ai_ws/"), f_type, f_idx, f_bit)


def sta_fault_campaign(f_bit_range = 1, f_bit_start = 62):
    campaign_results = {}

    golden_lib = stm_edgeai_lib.gen_lib()

    os.makedirs("./fault_campaign", exist_ok=True)
    os.makedirs("./fault_campaign/golden", exist_ok=True)

    os.system("rm -rf ./fault_campaign/*")
    os.system(f"cp -r {golden_lib} ./fault_campaign/golden/")

    with ProcessPoolExecutor(max_workers=n_th) as executor:
        futures = []
        weights = stm_edgeai_lib.weights_parser()

        for f_type in ["sta0", "sta1"]:
            campaign_results[f_type] = {}
            for f_idx in range(len(weights)):
                campaign_results[f_type][f_idx] = {}
                for f_bit in range(f_bit_start, f_bit_start-f_bit_range, -1):
                    campaign_results[f_type][f_idx][f_bit] = None

                    futures.append(executor.submit(simulate_fault, weights.copy(), f_type, f_idx, f_bit))

        for future in as_completed(futures):
            try:
                result, f_type, f_idx, f_bit = future.result()
                campaign_results[f_type][f_idx][f_bit] = result
            except Exception as e:
                print(f"Error occurred: {e}")

    return campaign_results

result = sta_fault_campaign()

with open("out_dict.txt", 'w') as f:
    f.write(str(result))