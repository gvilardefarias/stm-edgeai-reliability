from tracemalloc import start
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stm_edgeai_lib as stm
import fault_campaign as FC
import graph_gen as gg

#in_file = "out_files/out_dict_16b.txt"
#in_file = "out_dict.txt"
#network_info = "./hardening/c_code_validete/gmp/tmr_voter/st_ai_ws/network_c_graph.json"
#weights_file = "./st_ai_output/src/network_data_params.c"
in_file = "out_files/gmp/out_dict_16bof32_dataset.txt"
network_info = "./hardening/c_code_validete/gmp/original/st_ai_ws/network_c_graph.json"
weights_file = "./hardening/c_code_validete/gmp/original/st_ai_output/src/network_data_params.c"

def dict_to_df_sta(data_dict):
    sta0 = pd.DataFrame.from_dict(data_dict['sta0'], orient='index')
    sta1 = pd.DataFrame.from_dict(data_dict['sta1'], orient='index')

    for b in sta0:
        for w in sta0.index:
            sta0.at[w, b] = float(sta0.at[w, b][:-1])
    for b in sta1:
        for w in sta1.index:
            sta1.at[w, b] = float(sta1.at[w, b][:-1])


    sta0= sta0.reset_index(names=['Weight'])
    sta0 = sta0.melt(id_vars=sta0.columns[0], var_name='Bit', value_name='Acc')
    sta0 = sta0.set_index(['Weight','Bit'])
    sta1 = sta1.reset_index(names=['Weight'])
    sta1 = sta1.melt(id_vars=sta1.columns[0], var_name='Bit', value_name='Acc')
    sta1 = sta1.set_index(['Weight','Bit'])

    df = pd.DataFrame()
    df['sta1'] = sta1
    df['sta0'] = sta0
    df = df.reset_index().sort_values(['Weight', 'Bit']).reset_index(drop=True)
    df = df.melt(id_vars=['Weight', 'Bit'], var_name='Fault Type', value_name='Acc')

    return df

def dict_to_df(data_dict):
    df = pd.DataFrame.from_dict(data_dict['bf'], orient='index')

    for b in df:
        for w in df.index:
            df.at[w, b] = float(df.at[w, b][:-1])


    df = df.reset_index(names=['Weight'])
    df = df.melt(id_vars=df.columns[0], var_name='Bit', value_name='Acc')
    #df = df.set_index(['Weight','Bit'])

    return df

def compute_per_layer_acc_sta(df, layers_info):
    acc_drop = []

    for layer in layers_info:
        start_byte = layer['offset']
        end_byte = layer['size'] + start_byte

        start_weight = start_byte // 8
        end_weight = end_byte // 8

        start_bit = (start_byte % 8) * 8
        end_bit = (end_byte % 8) * 8

        layer_df = df[((df['Weight'] >= start_weight) & (df['Bit'] >= start_bit)) & 
                      ((df['Weight'] < end_weight) | (df['Weight'] == end_weight & (df['Bit'] < end_bit)))]

        df.loc[((df['Weight'] >= start_weight) & (df['Bit'] >= start_bit)) & 
                      ((df['Weight'] < end_weight) | (df['Weight'] == end_weight & (df['Bit'] < end_bit))), "layer_name"] = layer["buffer_name"]
        
        layer_acc_sta0 = layer_df[layer_df['Fault Type'] == 'sta0']['Acc'].mean()
        layer_acc_sta1 = layer_df[layer_df['Fault Type'] == 'sta1']['Acc'].mean()
        layer_acc = layer_df['Acc'].mean()
        layer_std = layer_df['Acc'].std()
        layer['reliability'] = {'accuracy_drop': {'sta0': 100 - layer_acc_sta0, 'sta1': 100 - layer_acc_sta1, 'mean': 100 - layer_acc, 'std': layer_std}}

        acc_drop.append({
            'layer_name': layer['buffer_name'],
            'sta0': 100 - layer_acc_sta0,
            'sta1': 100 - layer_acc_sta1,
            'mean': 100 - layer_acc,
            'std': layer_std
        })  

    layers_df = pd.DataFrame(acc_drop)
    return layers_df

def compute_per_layer_acc(df, layers_info):
    acc_drop = []

    for layer in layers_info:
        start_byte = layer['offset']
        end_byte = layer['size'] + start_byte

        start_weight = start_byte // 8
        end_weight = end_byte // 8

        start_bit = (start_byte % 8) * 8
        end_bit = (end_byte % 8) * 8

        layer_df = df[((df['Weight'] >= start_weight) & (df['Bit'] >= start_bit)) & 
                      ((df['Weight'] < end_weight) | (df['Weight'] == end_weight & (df['Bit'] < end_bit)))]

        df.loc[((df['Weight'] >= start_weight) & (df['Bit'] >= start_bit)) & 
                      ((df['Weight'] < end_weight) | (df['Weight'] == end_weight & (df['Bit'] < end_bit))), "layer_name"] = layer["buffer_name"]
        
        layer_acc = layer_df['Acc'].mean()
        layer_acc_drop_norm = (100 - layer_df['Acc']).sum() / df['Acc'].count()
        layer_std = layer_df['Acc'].std()
        layer['reliability'] = {'accuracy_drop': {'mean': 100 - layer_acc, 'std': layer_std, 'norm': layer_acc_drop_norm}}

        acc_drop.append({
            'layer_name': layer['buffer_name'],
            'mean': 100 - layer_acc,
            'std': layer_std,
            'norm': layer_acc_drop_norm
        })  

    layers_df = pd.DataFrame(acc_drop)
    print(layers_df)
    print(f"Overall accuracy drop: {layers_df['norm'].sum()}")
    return layers_df

def plot_per_layer_acc(data_dict):
    df = dict_to_df(data_dict)

    # TODO fix when we are checking a model that was not compiled
    # this path is hard coded in the lib
    layers_info  = stm.get_layers_info(network_info)
    layers_df = compute_per_layer_acc(df, layers_info)

    stm.set_layers_info(layers_info)

    #gg.per_layer_sta_bd(layers_df)
    gg.per_layer_sta_ov(layers_df)

def get_unsimulated_faults(data_dict):
    unsimulated_faults = []
    for fault_type in data_dict:
        for weight in data_dict[fault_type]:
            for bit in data_dict[fault_type][weight]:
                if data_dict[fault_type][weight][bit] == None:
                    unsimulated_faults.append((fault_type, weight, bit))
    
    return unsimulated_faults

def sta_to_bf(data_dict, weights_file):
    bf = {'bf':{}}
    weights = stm.weights_parser(weights_file)

    for fault_type in data_dict:
        for weight in data_dict[fault_type]:
            for bit in data_dict[fault_type][weight]:
                if FC.inject_sta_fault(weights, fault_type, weight, bit)[1]:
                    if weight in bf['bf']:
                        bf['bf'][weight][bit] = data_dict[fault_type][weight][bit]
                    else:
                        bf['bf'][weight] = {bit:data_dict[fault_type][weight][bit]}
                    
    return bf

if __name__ == "__main__":
    with open(in_file, 'r') as f:
        data = f.read()
        data_dict = eval(data)

    data_dict = sta_to_bf(data_dict, weights_file)
    plot_per_layer_acc(data_dict)
#    unsimulated_faults = get_unsimulated_faults(data_dict)