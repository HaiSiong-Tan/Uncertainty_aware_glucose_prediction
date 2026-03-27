# configs.py
# Configuration file for glucose prediction project

# =========================
# Data settings
# =========================
DATA_DIR = "./Preprocessed"

PATIENT_IDS = Patient_ID = ["01", "02", "03", "04", "05", "06", "07", "09", "10", "11", "14", "15",
                   "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]

FEATURES = ["glucose", "heart_rate", "bolus", "carb"]

COLUMN_MAP = {
    "glucose": 0,
    "heart_rate": 1,
    "calories": 2,
    "steps": 3,
    "basal_rate": 4,
    "bolus": 5,
    "carb": 6,
}


HORIZON_LENGTH = 6

# =========================
# Training settings
# =========================
BATCH_SIZE = 1024
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
LAMBDA_REG = 0.01
REG_TERM = "kl"

# =========================
# Model settings
# =========================
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
MAX_LEN = 100

# =========================
# Paths
# =========================
MODEL_SAVE_PATH = "./H_S_transf_evid_kl_model.pth"
LOSS_SAVE_PATH = "./transformer_training_loss_plot.png"
ECP_GRAPH_SAVE_PATH = "./ecp_graph_plot.png"
