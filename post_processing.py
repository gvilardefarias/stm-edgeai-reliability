import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

df = dict_to_df(data_dict)

pivot = df.pivot_table(index=['Weight', 'Bit'], columns='Fault Type', values='Acc').reset_index()
pivot['label'] = pivot['Weight'].astype(str) + ':' + pivot['Bit'].astype(str)
plot_df = pivot.set_index('label')[['sta0', 'sta1']]

ax = plot_df.plot(kind='bar', figsize=(12, 4))
ax.set_xlabel('Weight:Bit')
ax.set_ylabel('Acc')
ax.legend(title='Fault Type', fontsize=10)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#ax = df.plot(kind='bar', y = 'Fault Type', x = 'Weight')
#plt.rcParams.update({'font.size': 16})
#plt.xlabel('Weight Index and Bit Position', fontsize=18)
#plt.ylabel('Accuracy (%)', fontsize=18)
#ax.legend(fontsize=18) 
#ax.tick_params(axis='x', labelsize=16, rotation=45)
#ax.tick_params(axis='y', labelsize=16)
#plt.show()