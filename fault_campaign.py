from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import stm_edgeai_lib
from contextlib import contextmanager

n_th = 32

def inject_sta_fault(data, f_type, f_idx, f_bit):
    prev_value = data[f_idx]

    if f_type == "sta0":
        data[f_idx] &= ~(1 << f_bit)
    elif f_type == "sta1":
        data[f_idx] |= (1 << f_bit)
    
    return (data, prev_value != data[f_idx])


def simulate_fault(weights, f_type, f_idx, f_bit, remove_files = True):
    f_id = f"{f_type}_w{f_idx}_b{f_bit}"

    work_path = os.getcwd() + f"/fault_campaign/{f_id}/"
    lib_path = f"{work_path}inspector_network/workspace/"

    os.makedirs(f"./fault_campaign/{f_id}", exist_ok=True)
    os.system(f"cp -r ./fault_campaign/golden/* ./fault_campaign/{f_id}/")

    faulty_weights, fault_injected = inject_sta_fault(weights.copy(), f_type, f_idx, f_bit)

    if fault_injected:
        os.system(f"rm {lib_path}generated/network_data_params.c")
        stm_edgeai_lib.weights_file_gen(faulty_weights, out_file=f"{lib_path}generated/network_data_params.c", weights_file=f"{work_path}../../st_ai_output/src/network_data_params.c")
        stm_edgeai_lib.compile_lib(lib_path)
        stm_edgeai_lib.validade_lib(lib_path, exec_path=work_path)
        accuracy = stm_edgeai_lib.get_x_cross_accuracy(f"{work_path}st_ai_ws/")
    else:
        os.system(f"cp -r ./st_ai_ws ./fault_campaign/{f_id}/")
        accuracy = stm_edgeai_lib.get_x_cross_accuracy(f"{work_path}st_ai_ws/")

    if remove_files:
        os.system(f"rm -rf {work_path}")
    
    return (accuracy, f_type, f_idx, f_bit)

#def gen_f_bit_positions(f_bit_range = 16, f_bit_start = 62, f_bit_step = 32):
def gen_f_bit_positions(f_bit_range = 3, f_bit_start = 62, f_bit_step = 32):
    f_bit_positions = []
    for start in range(f_bit_start, -1, -f_bit_step):
        f_bit_positions.extend(list(range(start, start - f_bit_range, -1)))
    return f_bit_positions

def sta_fault_campaign(f_bit_positions = gen_f_bit_positions(), remove_files = True):
    campaign_results = {}

    golden_lib = stm_edgeai_lib.gen_lib()

    os.makedirs("./fault_campaign", exist_ok=True)
    os.makedirs("./fault_campaign/golden", exist_ok=True)

    os.system("rm -rf ./fault_campaign/*")
    os.system(f"cp -r {golden_lib} ./fault_campaign/golden/")

#    with ProcessPoolExecutor(max_workers=n_th) as executor:
    with ProcessPoolExecutor() as executor:
        futures = []
        weights = stm_edgeai_lib.weights_parser()

        for f_type in ["sta0", "sta1"]:
            campaign_results[f_type] = {}
            for f_idx in range(len(weights)):
                campaign_results[f_type][f_idx] = {}
                for f_bit in f_bit_positions:
                    campaign_results[f_type][f_idx][f_bit] = None

                    futures.append(executor.submit(simulate_fault, weights.copy(), f_type, f_idx, f_bit, remove_files))

        for future in as_completed(futures):
            try:
                result, f_type, f_idx, f_bit = future.result()
                campaign_results[f_type][f_idx][f_bit] = result
            except Exception as e:
                print(f"Error occurred: {e}")

    return campaign_results

start = time.time()
result = sta_fault_campaign()
end = time.time()
time_taken = end - start

print("Fault injection campaign completed.")
print(f"Time taken: {time_taken:.2f} seconds")

with open("out_dict.txt", 'w') as f:
    f.write(str(result))