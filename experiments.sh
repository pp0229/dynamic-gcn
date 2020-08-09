


echo "-------------------------------------------"
echo "ATTENTION WEIGHTS"
echo "-------------------------------------------"




echo "TEST 1"
python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:1

echo "TEST 2"
python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence additive \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:1

echo "TEST 3"
python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type sequential --snapshot-num 5 --cuda cuda:1

echo "TEST 4"
python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence dot_product \
    --dataset-name Twitter15 --dataset-type temporal --snapshot-num 5 --cuda cuda:1





# echo "TEST 1"
# python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence additive \
#     --dataset-name Twitter16 --dataset-type sequential --snapshot-num 5 --cuda cuda:2

# echo "TEST 2"
# python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence additive \
#     --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 --cuda cuda:2

# echo "TEST 3"
# python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence dot_product \
#     --dataset-name Twitter16 --dataset-type sequential --snapshot-num 5 --cuda cuda:2

# echo "TEST 4"
# python ./dynamic-gcn/attention_main.py --model PrintAttention --learning-sequence dot_product \
#     --dataset-name Twitter16 --dataset-type temporal --snapshot-num 5 --cuda cuda:2







# Twitter16 - additive - sequential
# [0.5453795793758746, 0.18607971137285342, 0.12037973020923161, 0.0835181871034608, 0.0646427957831597]


# Twitter16 - additive - temporal
# [0.30465886592607877, 0.19248922342903044, 0.16711173941816604, 0.16996791050066273, 0.1657722630919654]



# Twitter16 - dot_product - sequential


# [0.09906308015024354, 0.10682040181968491, 0.13279768188043423, 0.17391251986300896, 0.48740631498304415]
# [0.08292262411257523, 0.09390404120905692, 0.1198509351813264, 0.1627792409189541, 0.540543157607317]
# [0.07508089698369547, 0.08530374527548906, 0.11067631515732818, 0.15409606172967477, 0.5748429815342397]
# [0.06975437415046493, 0.07972761388831895, 0.10325308157995094, 0.1475816463637215, 0.5996832863462193]
# [0.06521714759055511, 0.07508541430224225, 0.09692535132274065, 0.1395410362060018, 0.6232310520875969]


# Twitter16 - dot_product - temporal

# [0.12956889080999234, 0.1466957282994313, 0.17249690700118347, 0.19539642835516385, 0.35584204703753375]
# [0.12108390276155528, 0.1430785157011336, 0.17105935634131453, 0.1957102866272921, 0.36906794026310064]
# [0.11783121987786223, 0.13963658291204625, 0.16905873560112125, 0.19576820178152043, 0.3777052647445673]
# [0.11586449836263102, 0.13770894765440994, 0.16683727275458748, 0.1933741121207792, 0.38621517119032367]
# [0.11008057673889239, 0.13307989516753765, 0.16218012007820629, 0.188888223000843, 0.4057711870122103]

