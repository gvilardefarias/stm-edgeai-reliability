import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

in_file = "out_dict.txt"

with open(in_file, 'r') as f:
    data = f.read()
    data_dict = eval(data)


sta0 = pd.DataFrame.from_dict(data_dict['sta0'], orient='index')
sta1 = pd.DataFrame.from_dict(data_dict['sta1'], orient='index')

for b in sta0:
    for w in sta0.index:
        sta0.at[w, b] = float(sta0.at[w, b][:-1])
for b in sta1:
    for w in sta1.index:
        sta1.at[w, b] = float(sta1.at[w, b][:-1])


df = pd.DataFrame()
df['sta0'] = sta0.stack()
df['sta1'] = sta1.stack()

df = (df.loc[(df['sta0'].index.get_level_values(0) < 40)])

ax = df.plot(kind='bar')
plt.rcParams.update({'font.size': 16})
plt.xlabel('Weight Index and Bit Position', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
ax.legend(fontsize=18) 
ax.tick_params(axis='x', labelsize=16, rotation=45)
ax.tick_params(axis='y', labelsize=16)
plt.show()