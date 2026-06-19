# Deep Residual Fingerprinting

Official repository for the paper **"Deep Residual Fingerprinting: A Self-Supervised Spatio-Temporal Transformer for Network Anomaly Detection"** by **Zilan Wang, Rongxing Lu, and Mohammad Zulkernine**.

This code implements a self-supervised spatio-temporal Transformer framework that learns normal network dynamics exclusively from benign traffic through multi-modal next-packet prediction of semantic embeddings, inter-arrival times, and packet lengths. Prediction residuals are distilled into a compact 12-dimensional statistical fingerprint and classified via Isolation Forests at early observation depths (10, 20, 50 packets) and upon flow termination.

## Setup

```bash
conda env create -f environment.yml
conda activate deep_residual
```

## Data

Download CIC-IDS-2017 from <https://www.unb.ca/cic/datasets/ids-2017.html>. Place the labelled flow CSV files and raw pcap files as follows:

```txt
CIC-IDS-2017/
├── GeneratedLabelledFlows/
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── raw_data_pcap/
    ├── Monday-WorkingHours.pcap
    ├── Tuesday-WorkingHours.pcap
    ├── Wednesday-workingHours.pcap
    ├── Thursday-WorkingHours.pcap
    └── Friday-WorkingHours.pcap
```

## Preprocessing

`preprocess.py` calls two external system commands: `editcap` trims the raw pcap to the flow's time window, and `tcpdump` applies the flow-level packet filter. Make sure both commands are available before preprocessing.

For each CIC-IDS-2017 subset, set the CSV and pcap paths at the top of `preprocess.py`:

```py
CSV_FILE = "CIC-IDS-2017/GeneratedLabelledFlows/Monday-WorkingHours.pcap_ISCX.csv"
PCAP_FILE = "CIC-IDS-2017/raw_data_pcap/Monday-WorkingHours.pcap"
```

Run preprocessing from the repository root:

```bash
python preprocess.py
```

The output is written to `preprocessed/<CSV_NAME>_maxpkts50_payload216/chunk_*.pkl`. Ensure that these directory names match `TRAIN_DIR` and `TEST_DIRS` in `eval/config.py`.

## Training

Run training from `train/`:

```bash
cd train
python train.py
```

Before training, check the Monday benign training path:

```py
TRAIN_DIR = "../preprocessed/Monday-WorkingHours_maxpkts50_payload216"
```

`VALID_DIR` may point to any preprocessed attack subset. It is used only for checkpoint-time validation: use the current Transformer checkpoint to obtain the residual features of Monday benign flows and an Isolation Forest is fitted, and AUC is reported against the validation set.

```py
VALID_DIR = "../preprocessed/Friday-WorkingHours-Afternoon-DDos_maxpkts50_payload216"
```

Model checkpoints are saved under `checkpoint/`.

## Evaluation

Change to `eval/` directory to run evaluation on the datasets from Tuesday to Friday:

```bash
cd eval
```

Set the Transformer checkpoint path in `eval/config.py`. We have provided a trained checkpoint as:

```py
MODEL_PATH = "../checkpoint/traffic_transformer_multihead.pth"
```

Also check `TRAIN_DIR`. During evaluation, Monday benign flows are split into two parts: one for fitting checkpoint-specific Isolation Forests, and the other for end-to-end threshold calibration to control the overall false positive rate.

Run the evaluation script:

```bash
python eval.py
```

Main options in `eval/config.py`:

```py
TRAIN_DIR = "../preprocessed/Monday-WorkingHours_maxpkts50_payload216"
TEST_DIRS = [...]
CHECKPOINTS = [10, 20, 50]
TARGET_E2E_FPR = 0.05
```

The script reports F1, TPR, FPR, average alert packet index, and observation saving.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.