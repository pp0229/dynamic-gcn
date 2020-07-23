import matplotlib.pyplot as plt

fig, ax = plt.subplots()

import matplotlib.pyplot as plt
import numpy as np

# a = np.random.random((16, 16))


array = [[0.0491, 0.1027, 0.3560, 0.0658, 0.4263],
        [0.0311, 0.1233, 0.3419, 0.0448, 0.4589],
        [0.0187, 0.0592, 0.4126, 0.0308, 0.4788],
        [0.0385, 0.0866, 0.3439, 0.0782, 0.4528],
        [0.0155, 0.0550, 0.3318, 0.0281, 0.5695]]

array = [[0.1499, 0.1520, 0.2086, 0.2193, 0.2702],
        [0.0979, 0.1298, 0.2090, 0.2442, 0.3191],
        [0.0638, 0.0991, 0.2035, 0.2661, 0.3675],
        [0.0470, 0.0812, 0.1865, 0.2805, 0.4050],
        [0.0397, 0.0727, 0.1764, 0.2775, 0.4337]]



array = np.array(array)
plt.imshow(array, cmap='YlGnBu')
plt.colorbar()

for y in range(array.shape[0]):
    for x in range(array.shape[1]):
        # plt.text(x + 0.25, y + 0.25, '%.4f' % array[y, x],
        #     horizontalalignment='center',
        #     verticalalignment='center',
        # )
        plt.text(x, y, '%.4f' % array[y, x],
            horizontalalignment='center',
            verticalalignment='center',
        )


plt.savefig("./heatmap.png")

"""
import seaborn as sns

sns.heatmap(array, annot=True)
plt.savefig("./heatmap2.png")
"""