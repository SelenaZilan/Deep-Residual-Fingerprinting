BATCH_SIZE = 32
CHECKPOINTS = [10, 20, 50]
TARGET_E2E_FPR = 0.05
PERCENTILE_SEARCH_LOW = 50.0
PERCENTILE_SEARCH_HIGH = 99.99
PERCENTILE_SEARCH_STEPS = 25
MONDAY_CALIB_LIMIT = 10000
MONDAY_TRAIN_LIMIT = 30000
DEVICE = "auto"

MODEL_PATH = "../checkpoint/traffic_transformer_multihead.pth"
TRAIN_DIR = "../preprocessed/Monday-WorkingHours_maxpkts50_payload216"
TEST_DIRS = [
    "../preprocessed/Friday-WorkingHours-Afternoon-DDos_maxpkts50_payload216",
    "../preprocessed/Friday-WorkingHours-Afternoon-PortScan_maxpkts50_payload216",
    "../preprocessed/Friday-WorkingHours-Morning_maxpkts50_payload216",
    "../preprocessed/Tuesday-WorkingHours_maxpkts50_payload216",
    "../preprocessed/Wednesday-workingHours_maxpkts50_payload216",
    "../preprocessed/Thursday-WorkingHours-Afternoon-Infilteration_maxpkts50_payload216",
    "../preprocessed/Thursday-WorkingHours-Morning-WebAttacks_maxpkts50_payload216",
]

ATTACK_NAME_MAP = {
    "Friday-WorkingHours-Afternoon-DDos": "DDoS",
    "Friday-WorkingHours-Afternoon-PortScan": "PortScan",
    "Friday-WorkingHours-Morning": "Bot",
    "Tuesday-WorkingHours": "Brute Force",
    "Wednesday-workingHours": "DoS",
    "Thursday-WorkingHours-Afternoon-Infilteration": "Infiltration",
    "Thursday-WorkingHours-Morning-WebAttacks": "Web Attack",
}

ATTACK_DAY_MAP = {
    "DDoS": "Fri",
    "PortScan": "Fri",
    "Bot": "Fri",
    "Brute Force": "Tue",
    "DoS": "Wed",
    "Infiltration": "Thu",
    "Web Attack": "Thu",
}
