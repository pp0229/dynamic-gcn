# Dynamic GCN for Rumor Detection

Dynamic Graph Convolutional Networks with Attention Mechanism for Rumor Detection on Social Media (Public)

### Requirements

* Python, PIP 3.6
* CUDA 10.2
* PyTorch 1.5
* PyTorch Geometric 1.4
  + torch-scatter 2.0.4
  + torch-sparse 0.6.0

### Setup

``` bash
python -m venv env
source ./env/bin/activate

pip install --upgrade pip
pip install numpy
pip install torch

# PyTorch Geometric
#   https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
#   Requirements: torch-scatter, torch-sparse

CUDA=cu102
TORCH=1.5.0

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
```

### Overview

* Snapshot Generation

<!-- > ![](https://github.com/JihoChoi/dynamic-gcn/blob/master/assets/model.png?raw=true) -->

* Model Overview

<!-- > ![](https://github.com/JihoChoi/dynamic-gcn/blob/master/assets/model.png?raw=true) -->

### Project Structure

``` markdown
├── README.md
├── prepare_dataset.sh
├── resources
│   ├── Twitter15_label_All.txt
│   ├── Twitter16_label_All.txt
│   └── Weibo_label_All.txt
│
├── dynamic-gcn (scripts)
│   ├── utils.py
│   ├── preparation
│   │   ├── dataset_preprocess.py
│   │   ├── snapshot_preparation.py
│   │   ├── dataset_preparation.py
│   ├── tools


<!-- TODO -->
│   ├── dataset

│   ├── project_settings.py
│   ├── proof-of-concept
│   │   └── snapshots_visualization.py  <-
│   ├── preparation
│   │   ├── dataset_preprocess.py       <- add temporal information
│   │   ├── dataset_preparation.py      <- data to graph datasets
│   │   ├── random-folds.py
│   │   └── dataset_validation.py       <- deprecated
│   ├── main.py
│   ├── dataset.py
│   ├── models.py
│   ├── layers.py
│   └── utils.py
└── baselines
    ├── BiGCN
    └── ...

```

### Usages

##### Prepare Dataset

``` bash
python ./scripts/data-preparation/validate_dataset.py
```

##### Test & Inference

``` bash
python main.py
```
