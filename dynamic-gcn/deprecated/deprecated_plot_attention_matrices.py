import matplotlib
matplotlib.use('pdf')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json


# Twitter16 - additive - sequential
data_1 = [[0.5453795793758746, 0, 0, 0, 0],
        [0, 0.18607971137285342, 0, 0, 0],
        [0, 0, 0.12037973020923161, 0, 0],
        [0, 0, 0, 0.0835181871034608, 0],
        [0, 0, 0, 0, 0.0646427957831597]]

# Twitter16 - additive - temporal
data_2 = [[0.30465886592607877, 0, 0, 0, 0],
        [0, 0.19248922342903044, 0, 0, 0],
        [0, 0, 0.16711173941816604, 0, 0],
        [0, 0, 0, 0.16996791050066273, 0],
        [0, 0, 0, 0, 0.1657722630919654]]

# Twitter16 - dot_product - sequential
data_3 = [[0.09906308015024354, 0.10682040181968491, 0.13279768188043423, 0.17391251986300896, 0.48740631498304415],
        [0.08292262411257523, 0.09390404120905692, 0.1198509351813264, 0.1627792409189541, 0.540543157607317],
        [0.07508089698369547, 0.08530374527548906, 0.11067631515732818, 0.15409606172967477, 0.5748429815342397],
        [0.06975437415046493, 0.07972761388831895, 0.10325308157995094, 0.1475816463637215, 0.5996832863462193],
        [0.06521714759055511, 0.07508541430224225, 0.09692535132274065, 0.1395410362060018, 0.6232310520875969]]

# Twitter16 - dot_product - temporal
data_4 = [[0.12956889080999234, 0.1466957282994313, 0.17249690700118347, 0.19539642835516385, 0.35584204703753375],
    [0.12108390276155528, 0.1430785157011336, 0.17105935634131453, 0.1957102866272921, 0.36906794026310064],
    [0.11783121987786223, 0.13963658291204625, 0.16905873560112125, 0.19576820178152043, 0.3777052647445673],
    [0.11586449836263102, 0.13770894765440994, 0.16683727275458748, 0.1933741121207792, 0.38621517119032367],
    [0.11008057673889239, 0.13307989516753765, 0.16218012007820629, 0.188888223000843, 0.4057711870122103]]

array_1 = np.array(data_1)
array_2 = np.array(data_2)
array_3 = np.array(data_3)
array_4 = np.array(data_4)


# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

im1 = ax1.imshow(array_1, cmap='GnBu', vmin=0, vmax=0.6)
im2 = ax2.imshow(array_2, cmap='GnBu', vmin=0, vmax=0.6)
im3 = ax3.imshow(array_3, cmap='GnBu', vmin=0, vmax=0.7)
im4 = ax4.imshow(array_4, cmap='GnBu', vmin=0, vmax=0.7)


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.25])
fig.colorbar(im2, cax=cbar_ax)

cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.25])
fig.colorbar(im4, cax=cbar_ax)

ax3.set_xlabel('Sequential Snapshots')
ax4.set_xlabel('Temporal Snapshots')


array = array_1
for y in range(array.shape[0]):
    for x in range(array.shape[1]):
        ax1.text(x, y, '%.3f' % array[y, x],
            horizontalalignment='center',
            verticalalignment='center',
        )

array = array_2
for y in range(array.shape[0]):
    for x in range(array.shape[1]):
        ax2.text(x, y, '%.3f' % array[y, x],
            horizontalalignment='center',
            verticalalignment='center',
        )

array = array_3
for y in range(array.shape[0]):
    for x in range(array.shape[1]):
        ax3.text(x, y, '%.3f' % array[y, x],
            horizontalalignment='center',
            verticalalignment='center',
        )

array = array_4
for y in range(array.shape[0]):
    for x in range(array.shape[1]):
        ax4.text(x, y, '%.3f' % array[y, x],
            horizontalalignment='center',
            verticalalignment='center',
        )




fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [left, bottom, width, height]
cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
fig.colorbar(im4, cax=cbar_ax)
# fig.colorbar(im, cax=cbar_ax, norm=mpl.colors.Normalize(vmin=0.25, vmax=0.75))



plt.savefig("./twitter16_attention_combined.png", bbox_inches='tight')
plt.clf()

