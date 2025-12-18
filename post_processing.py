from tracemalloc import start
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stm_edgeai_lib as stm
import graph_gen as gg

in_file = "out_dict.txt"

with open(in_file, 'r') as f:
    data = f.read()
    data_dict = eval(data)

def dict_to_df(data_dict):
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

    acc_drop_df = pd.DataFrame(acc_drop)
    return acc_drop_df

df = dict_to_df(data_dict)

layers_info  = stm.get_layers_info()
acc_drop_df = compute_per_layer_acc(df, layers_info)

stm.set_layers_info(layers_info)

print(acc_drop_df)
#gg.per_layer_sta_bd(acc_drop_df)
gg.per_layer_sta_ov(acc_drop_df)