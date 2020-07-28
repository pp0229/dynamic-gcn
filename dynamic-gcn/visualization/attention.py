import matplotlib.pyplot as plt
import numpy as np
import json

fig, ax = plt.subplots()

# a = np.random.random((16, 16))

def load_json_file(path):
    with open(path, "r") as json_file:
        data = json.loads(json_file.read())
    return data

# data = load_json_file("./attention_2020_0727_0105.json")


# Twitter16 - sequential - additive
# file_data = open("./attention_2020_0727_0105.txt", 'r')
# [0.5453795793758746, 0.18607971137285342, 0.12037973020923161, 0.0835181871034608, 0.0646427957831597]

# Twitter16 - temporal - additive
# file_data = open("./attention_2020_0727_0122.txt", 'r')
# [0.30465886592607877, 0.19248922342903044, 0.16711173941816604, 0.16996791050066273, 0.1657722630919654]


# -------------------------
# Additive
# -------------------------

"""
mean_score = [0, 0, 0, 0, 0]
count = 0

for line in file_data:
    data = eval(line)

    for score in data:

        mean_score = [x + y for x, y in zip(mean_score, score)]
        count += 1
        # print(mean_score)

print(count, mean_score)
result = [score / count for score in mean_score]
print(result)
print(sum(result))
"""


"""

# file_data = open("./attention_2020_0727_0147.txt", 'r')
file_data = open("./attention_2020_0727_0203.txt", 'r')


mean_score = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]

count = 0

for line in file_data:
    data = eval(line)
    for score in data:
        mean_score[0] = [x + y for x, y in zip(mean_score[0], score[0])]
        mean_score[1] = [x + y for x, y in zip(mean_score[1], score[1])]
        mean_score[2] = [x + y for x, y in zip(mean_score[2], score[2])]
        mean_score[3] = [x + y for x, y in zip(mean_score[3], score[3])]
        mean_score[4] = [x + y for x, y in zip(mean_score[4], score[4])]
        count += 1

print(count, mean_score)

result = []
result.append([score / count for score in mean_score[0]])
result.append([score / count for score in mean_score[1]])
result.append([score / count for score in mean_score[2]])
result.append([score / count for score in mean_score[3]])
result.append([score / count for score in mean_score[4]])

print("----------------------")
print(result[0])
print(result[1])
print(result[2])
print(result[3])
print(result[4])
"""





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


def save_figure(path, data):
    array = np.array(data)
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


    plt.savefig(path)
    plt.clf()



# Twitter16 - additive - sequential
array = [[0.5453795793758746, 0, 0, 0, 0],
        [0, 0.18607971137285342, 0, 0, 0],
        [0, 0, 0.12037973020923161, 0, 0],
        [0, 0, 0, 0.0835181871034608, 0],
        [0, 0, 0, 0, 0.0646427957831597]]

save_figure("./heatmap_additive_sequential.png", array)

# Twitter16 - additive - temporal
array = [[0.30465886592607877, 0, 0, 0, 0],
        [0, 0.19248922342903044, 0, 0, 0],
        [0, 0, 0.16711173941816604, 0, 0],
        [0, 0, 0, 0.16996791050066273, 0],
        [0, 0, 0, 0, 0.1657722630919654]]
save_figure("./heatmap_additive_temporal.png", array)



# Twitter16 - dot_product - sequential


array = [[0.09906308015024354, 0.10682040181968491, 0.13279768188043423, 0.17391251986300896, 0.48740631498304415],
        [0.08292262411257523, 0.09390404120905692, 0.1198509351813264, 0.1627792409189541, 0.540543157607317],
        [0.07508089698369547, 0.08530374527548906, 0.11067631515732818, 0.15409606172967477, 0.5748429815342397],
        [0.06975437415046493, 0.07972761388831895, 0.10325308157995094, 0.1475816463637215, 0.5996832863462193],
        [0.06521714759055511, 0.07508541430224225, 0.09692535132274065, 0.1395410362060018, 0.6232310520875969]]
save_figure("./heatmap_dot_product_sequential.png", array)


# Twitter16 - dot_product - temporal

array = [[0.12956889080999234, 0.1466957282994313, 0.17249690700118347, 0.19539642835516385, 0.35584204703753375],
    [0.12108390276155528, 0.1430785157011336, 0.17105935634131453, 0.1957102866272921, 0.36906794026310064],
    [0.11783121987786223, 0.13963658291204625, 0.16905873560112125, 0.19576820178152043, 0.3777052647445673],
    [0.11586449836263102, 0.13770894765440994, 0.16683727275458748, 0.1933741121207792, 0.38621517119032367],
    [0.11008057673889239, 0.13307989516753765, 0.16218012007820629, 0.188888223000843, 0.4057711870122103]]
save_figure("./heatmap_dot_product_temporal.png", array)


"""
import seaborn as sns

sns.heatmap(array, annot=True)
plt.savefig("./heatmap2.png")
"""