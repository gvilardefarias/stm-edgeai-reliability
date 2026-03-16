from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import os
import time
import stm_edgeai_lib
import argparse
import post_processing

n_th = 128
WEIGHTS = None
golden_acc = "100.00%"
golden_report = None
save_report = False
continue_cmp = False

parser = argparse.ArgumentParser(description='Run STA fault injection campaign.')
parser.add_argument('--continue_cmp', action='store_true', help='Continue an existing campaign by simulating only the unsimulated faults.', default=False)
args = parser.parse_args()
continue_cmp = args.continue_cmp

def inject_sta_fault(data, f_type, f_idx, f_bit):
    prev_value = data[f_idx]

    if f_type == "sta0":
        data[f_idx] &= ~(1 << f_bit)
    elif f_type == "sta1":
        data[f_idx] |= (1 << f_bit)
    
    return (data, prev_value != data[f_idx])


def simulate_fault(f_type, f_idx, f_bit, remove_files=True):
    f_id = f"{f_type}_w{f_idx}_b{f_bit}"

    work_path = os.path.abspath(f"./fault_campaign/{f_id}") + "/"
    lib_path = f"{work_path}inspector_network/workspace/"

    # BUG4 FIX: use absolute path for golden source to avoid relative path
    # race conditions across parallel workers
    golden_abs = os.path.abspath("./fault_campaign/golden")

    os.makedirs(f"./fault_campaign/{f_id}", exist_ok=True)
    os.system(f"cp -r {golden_abs}/. ./fault_campaign/{f_id}/")

    faulty_weights, fault_injected = inject_sta_fault(WEIGHTS.copy(), f_type, f_idx, f_bit)

    if fault_injected:
        os.system(f"rm {lib_path}generated/network_data_params.c")
        stm_edgeai_lib.weights_file_gen(
            faulty_weights,
            out_file=f"{lib_path}generated/network_data_params.c",
            weights_file=f"{work_path}../../st_ai_output/src/network_data_params.c"
        )
        stm_edgeai_lib.compile_lib(lib_path)
        stm_edgeai_lib.validade_lib(lib_path, exec_path=work_path)

        # BUG3 FIX: stedgeai writes network_report.json to the exec_path (work_path),
        # not to work_path/st_ai_ws/
        accuracy, report = stm_edgeai_lib.get_x_cross_accuracy(work_path)
    else:
        accuracy, report = golden_acc, golden_report

    if remove_files:
        os.system(f"rm -rf {work_path}")
    elif not fault_injected:
        os.system(f"cp -r ./st_ai_ws ./fault_campaign/{f_id}/")
    
    return (accuracy, report, f_type, f_idx, f_bit)

#def gen_f_bit_positions(f_bit_range = 16, f_bit_start = 62, f_bit_step = 32):
#def gen_f_bit_positions(f_bit_range = 2, f_bit_start = 62, f_bit_step = 32):
def gen_f_bit_positions(f_bit_range=16, f_bit_start=63, f_bit_step=32):
#def gen_f_bit_positions(f_bit_range = 16, f_bit_start = 63, f_bit_step = 16):
    f_bit_positions = []
    for start in range(f_bit_start, -1, -f_bit_step):
        f_bit_positions.extend(list(range(start, start - f_bit_range, -1)))
    return f_bit_positions

def sta_fault_campaign(f_bit_positions=gen_f_bit_positions(), remove_files=True, save_report=False):
    global WEIGHTS, golden_acc, golden_report
    campaign_results = {}
    report_results = {}

    golden_lib = stm_edgeai_lib.gen_lib()

    # BUG2 FIX: gen_lib() runs stedgeai validate in host mode (no --mode target),
    # which writes network_report.json into ./st_ai_ws/ in the cwd.
    # We must read the golden accuracy HERE, before rm -rf wipes fault_campaign/*
    # and before any further file manipulation can disturb the path.
    golden_acc, golden_report = stm_edgeai_lib.get_x_cross_accuracy(f"{golden_lib}/")

    os.makedirs("./fault_campaign", exist_ok=True)
    os.makedirs("./fault_campaign/golden", exist_ok=True)

    os.system("rm -rf ./fault_campaign/*")
    os.system(f"cp -r {golden_lib} ./fault_campaign/golden/")

    runing_futures = set()
    WEIGHTS = stm_edgeai_lib.weights_parser()

#    with ProcessPoolExecutor(max_workers=n_th) as executor:
    with ProcessPoolExecutor() as executor:
        for f_type in ["sta0", "sta1"]:
            campaign_results[f_type] = {}
            report_results[f_type] = {}
            for f_idx in range(len(WEIGHTS)):
                campaign_results[f_type][f_idx] = {}
                report_results[f_type][f_idx] = {}
                for f_bit in f_bit_positions:
                    campaign_results[f_type][f_idx][f_bit] = None
                    report_results[f_type][f_idx][f_bit] = None

                    if len(runing_futures) >= 2 * n_th:
                        done, runing_futures = wait(runing_futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                acc_result, report, f_type_r, f_idx_r, f_bit_r = future.result()
                                campaign_results[f_type_r][f_idx_r][f_bit_r] = acc_result
                                if save_report:
                                    report_results[f_type_r][f_idx_r][f_bit_r] = report
                            except Exception as e:
                                print(f"Error occurred: {e}")
                    
                    future = executor.submit(simulate_fault, f_type, f_idx, f_bit, remove_files)
                    runing_futures.add(future)

        # BUG5 FIX: use distinct variable names f_type_r/f_idx_r/f_bit_r to avoid
        # shadowing the outer loop variable f_type
        for future in as_completed(runing_futures):
            try:
                acc_result, report, f_type_r, f_idx_r, f_bit_r = future.result()
                campaign_results[f_type_r][f_idx_r][f_bit_r] = acc_result
                if save_report:
                    report_results[f_type_r][f_idx_r][f_bit_r] = report
            except Exception as e:
                print(f"Error occurred: {e}")

    return campaign_results, report_results

# TODO implement save_report
def continue_sta_fault_campaign(campaign_results, faults, remove_files=True, save_report=False):
    global WEIGHTS, golden_acc, golden_report
    report_results = {}

    # BUG2 FIX: same path correction as in sta_fault_campaign
    golden_acc, golden_report = stm_edgeai_lib.get_x_cross_accuracy("./fault_campaign/golden/st_ai_ws/")

    runing_futures = set()
    WEIGHTS = stm_edgeai_lib.weights_parser()

    #with ProcessPoolExecutor(max_workers=n_th) as executor:
    with ProcessPoolExecutor() as executor:
        for f_type, f_idx, f_bit in faults:
            if len(runing_futures) >= 2 * n_th:
                done, runing_futures = wait(runing_futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        acc_result, report, f_type_r, f_idx_r, f_bit_r = future.result()
                        campaign_results[f_type_r][f_idx_r][f_bit_r] = acc_result
                        if save_report:
                            report_results[f_type_r][f_idx_r][f_bit_r] = report
                    except Exception as e:
                        print(f"Error occurred: {e}")
            
            future = executor.submit(simulate_fault, f_type, f_idx, f_bit, remove_files)
            runing_futures.add(future)

        # BUG5 FIX: use distinct variable names to avoid shadowing
        for future in as_completed(runing_futures):
            try:
                acc_result, report, f_type_r, f_idx_r, f_bit_r = future.result()
                campaign_results[f_type_r][f_idx_r][f_bit_r] = acc_result
                if save_report:
                    report_results[f_type_r][f_idx_r][f_bit_r] = report
            except Exception as e:
                print(f"Error occurred: {e}")

    return campaign_results, report_results

if __name__ == "__main__":
    if not continue_cmp:
        start = time.time()
        result, report_results = sta_fault_campaign(save_report=save_report)
        end = time.time()
        time_taken = end - start
    
        print("Fault injection campaign completed.")
        print(f"Time taken: {time_taken:.2f} seconds")
    
        with open("out_dict.txt", 'w') as f:
            f.write(str(result))
    
        if save_report:
            with open("report_results.txt", 'w') as f:
                f.write(str(report_results))
    else:
        data_dict = None
        with open("out_dict.txt", 'r') as f:
            data = f.read()
            data_dict = eval(data)
        faults_to_simulate = post_processing.get_unsimulated_faults(data_dict)
        print(f"Continuing campaign with {len(faults_to_simulate)} faults to simulate.")
        start = time.time()
        result, report_results = continue_sta_fault_campaign(data_dict, faults_to_simulate, save_report=save_report)
        end = time.time()
        time_taken = end - start
        print("Fault injection campaign completed.")
        print(f"Time taken: {time_taken:.2f} seconds")

        with open("out_dict_new.txt", 'w') as f:
            f.write(str(result))
        
        if save_report:
            with open("report_results_new.txt", 'w') as f:
                f.write(str(report_results))