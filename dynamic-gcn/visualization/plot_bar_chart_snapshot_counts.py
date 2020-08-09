import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['hatch.linewidth'] = 0.25

plt.style.use('ggplot')

fig = plt.figure(figsize=(8, 4))


ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)



# DOT PRODUCT
twitter15_data = {
    'sequential_snapshots_acc': [0.813862, 0.821518, 0.819372, 0.818782],
    'sequential_snapshots_err': [0.003714052, .0022065, 0.002683005, 0.00217223],
    'temporal_snapshots_acc': [0.813862, 0.818518, 0.820554, 0.820938],
    'temporal_snapshots_err': [0.003714052, .002692152, 0.001768972, 0.001867656],
}

twitter16_data = {
    'sequential_snapshots_acc': [0.8039, 0.8332, 0.8294, 0.834104],
    'sequential_snapshots_err': [0.006333197, 0.005874756, 0.006668525, 0.008988481],
    'temporal_snapshots_acc': [0.8039, 0.8349, 0.8237, 0.828688],
    'temporal_snapshots_err': [0.006333197, 0.004497914, 0.005591217, 0.009678077],
}


def plot(ax, dataset_name, data):
    ax.grid(axis='x')
    opacity = 1  # 0.8
    bar_width = 0.25
    index = np.array([0, 1, 2, 3])
    labels = ['1', '2', '3', '5']
    ax.set_ylim(0.75, 0.9)
    ax.set_yticks(np.arange(0.75, 0.9, 0.05))

    # ax.set_ylim(0.8, 0.85)
    # ax.set_yticks(np.arange(0.8, 0.85, 0.05))


    rects1 = ax.bar(
        index,
        data['sequential_snapshots_acc'],
        bar_width,
        color='#84a3fa', alpha=opacity,
        hatch='++', edgecolor='white',
        label='sequential snapshots',
        yerr=data['sequential_snapshots_err'],
        error_kw=dict(lw=1, capsize=4, capthick=1, ecolor='#707070')
    )

    rects2 = ax.bar(
        index + bar_width + 0.05,
        data['temporal_snapshots_acc'],
        bar_width,
        color='#cf8282', alpha=opacity,
        hatch='//', edgecolor='white',
        label='temporal snapshots',
        yerr=data['temporal_snapshots_err'],
        error_kw=dict(lw=1, capsize=4, capthick=1, ecolor='#707070')
    )
    ax.set_title(dataset_name)
    ax.set_xticks(index + bar_width / 2 + 0.05 / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel('snapshot counts')
    # ax.set_ylabel('accuracy')
    if dataset_name == "Twitter 15":
        ax.set_ylabel('accuracy')
        ax.legend(bbox_to_anchor=(0, 0.8, 1, 0.2), loc="center", ncol=1, facecolor='white')
        ax.set_zorder(1)

"""B R G Y #84a3fa #cf8282 #83c2d5 #f8b27d"""


plot(ax1, "Twitter 15", twitter15_data)
plot(ax2, "Twitter 16", twitter16_data)
plt.savefig("./figures/bar_chart_01_one_column.png", bbox_inches='tight')
# plt.show()