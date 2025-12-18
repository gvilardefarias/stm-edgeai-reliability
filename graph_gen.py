import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

def per_layer_sta_ov(acc_drop_df):
    plot_mean_drop_df = acc_drop_df.set_index('layer_name')[['mean']]
    plot_std_df = acc_drop_df.set_index('layer_name')[['std']]

    plt.errorbar(plot_mean_drop_df.index, plot_mean_drop_df['mean'], yerr=plot_std_df['std'], fmt='o', ecolor='red', capsize=5)
    ax = plt.gca()

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Accuracy drop (%)')
    ax.set_title('Per-layer mean accuracy drop with standard deviation')
    ax.legend().set_visible(False)
    ax.set_ylim(0,100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()