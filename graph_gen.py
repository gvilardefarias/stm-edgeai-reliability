import matplotlib.pyplot as plt
import pandas as pd
import numpy as np#
import matplotlib as mpl
mpl.rcParams['font.size'] = 16


# Plot per-layer accuracy drop with bars for sta0 and sta1 with mean line
def per_layer_sta_bd(acc_drop_df):
    plot_acc_drop_df = acc_drop_df.set_index('layer_name')[['sta0', 'sta1']]
    plot_mean_drop_df = acc_drop_df.set_index('layer_name')[['mean']]

    ax = plot_acc_drop_df.plot(kind='bar', figsize=(10, 6), width=0.8)

    plot_mean_drop_df.plot(kind='line', marker='o', color='black', ax=ax)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy drop (%)')
    ax.set_title('Per-layer accuracy drop')
    ax.legend(title='Fault Type')
    ax.set_ylim(0,100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot per-layer mean accuracy drop with standard deviation
def per_layer_sta_ov(acc_drop_df):
    plot_mean_drop_df = acc_drop_df.set_index('layer_name')[['mean']]
    plot_std_df = acc_drop_df.set_index('layer_name')[['std']]
    
    for layer_name in plot_mean_drop_df.index:
        layer_name_short = layer_name.replace('_array','')
        plot_mean_drop_df = plot_mean_drop_df.rename(index={layer_name: layer_name_short})
        plot_std_df = plot_std_df.rename(index={layer_name: layer_name_short})

    plt.figure(figsize=(8, 6))
    plt.errorbar(plot_mean_drop_df.index, plot_mean_drop_df['mean'], yerr=plot_std_df['std'], fmt='o', ecolor='red', capsize=5)
    ax = plt.gca()

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Accuracy drop (%)')
    ax.set_title('Per-layer mean accuracy drop with standard deviation')
    ax.legend().set_visible(False)
    ax.set_ylim(0,25)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.2)
    plt.show()