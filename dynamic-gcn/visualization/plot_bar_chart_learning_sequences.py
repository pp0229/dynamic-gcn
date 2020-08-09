import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['hatch.linewidth'] = 0.25

plt.style.use('ggplot')

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)





twitter15_data = {
    'sequential_snapshots_acc': [0.761625, 0.775306667, 0.8177, 0.8194],
    'sequential_snapshots_err': [0.007742904, 0.002489185, 0.002780387, 0.002683005],
    'temporal_snapshots_acc': [0.76134, 0.776553333, 0.8267, 0.820554],
    'temporal_snapshots_err': [0.014852973, 0.006666237, 0.002248314, 0.001768972],
}

twitter16_data = {
    'sequential_snapshots_acc': [0.781362, 0.800476, 0.8278, 0.8294],
    'sequential_snapshots_err': [0.010368692, 0.005717818, 0.003292468, 0.006668525],
    'temporal_snapshots_acc': [0.778286, 0.79601, 0.8357872, 0.8237447],
    'temporal_snapshots_err': [0.00912148, 0.009288438, 0.00394949, 0.005591217],
}


def plot(ax, dataset_name, data):
    ax.grid(axis='x')
    opacity = 1  # 0.8
    bar_width = 0.25
    index = np.array([0, 1, 2, 3])
    labels = ['LSTM', 'GRU', 'additive', 'dot-product']
    ax.set_ylim(0.7, 0.9)
    ax.set_yticks(np.arange(0.7, 0.9, 0.05))

    rects1 = ax.bar(
        index,
        data['sequential_snapshots_acc'],
        bar_width,
        color='#83c2d5', alpha=opacity,
        hatch='++', edgecolor='white',
        label='sequential snapshots',
        yerr=data['sequential_snapshots_err'],
        error_kw=dict(lw=1, capsize=4, capthick=1, ecolor='#707070')
    )

    rects2 = ax.bar(
        index + bar_width + 0.05,
        data['temporal_snapshots_acc'],
        bar_width,
        color='#f8b27d', alpha=opacity,
        hatch='//', edgecolor='white',
        label='temporal snapshots',
        yerr=data['temporal_snapshots_err'],
        error_kw=dict(lw=1, capsize=4, capthick=1, ecolor='#707070')
    )
    ax.set_title(dataset_name)
    ax.set_xticks(index + bar_width / 2 + 0.05 / 2)
    ax.set_xticklabels(labels)
    # ax.set_xlabel('')
    # ax.set_ylabel('accuracy')
    if dataset_name == "Twitter 15":
        ax.set_ylabel('accuracy')
        ax.legend(bbox_to_anchor=(0, 0.8, 1, 0.2), loc="center", ncol=1, facecolor='white')
        ax.set_zorder(1)

"""B R G Y #84a3fa #cf8282 #83c2d5 #f8b27d"""

plot(ax1, "Twitter 15", twitter15_data)
plot(ax2, "Twitter 16", twitter16_data)
plt.savefig("./figures/bar_chart_02_one_column.png", bbox_inches='tight')
# plt.show()