from pathlib import Path


# ============================================================
# PROJEKTPFADE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

EXP2_ROOT = Path(r"D:\ComibnedData\Experiment2")
EXP3_ROOT = Path(r"D:\ComibnedData\Experiment3")

# Die vorhandenen Trainingsordner werden später intern
# in Training und Validation aufgeteilt.
EXP2_TRAIN_SOURCE = EXP2_ROOT / "train"
EXP3_TRAIN_SOURCE = EXP3_ROOT / "train"

# Die vorhandenen Validation-Ordner werden als unabhängige
# Testdatensätze verwendet.
EXP2_TEST_SOURCE = EXP2_ROOT / "validation"
EXP3_TEST_SOURCE = EXP3_ROOT / "validation"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUFBEREITETE DATENSÄTZE
# ============================================================

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.npz"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_data.npz"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.npz"

# ============================================================
# KLASSEN
# ============================================================

CLASS_NAMES = [
    "EMPTY",
    "FALL",
    "LYING",
    "SIT",
    "SIT-DOWN",
    "STAND",
    "STAND-UP",
    "WALK",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(CLASS_NAMES)
}

ID_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_ID.items()
}

NUM_CLASSES = len(CLASS_NAMES)


LABEL_ALIASES = {
    "EMPTY": "EMPTY",
    "LYING": "LYING",
    "LAYING": "LYING",

    "SIT": "SIT",

    "STAND": "STAND",
    "STANDING": "STAND",

    "WALK": "WALK",
    "WALKING": "WALK",

    "FALL": "FALL",

    "SITDOWN": "SIT-DOWN",
    "SIT-DOWN": "SIT-DOWN",

    "STANDUP": "STAND-UP",
    "STAND-UP": "STAND-UP",
}

# ============================================================
# DATENFORM
# ============================================================

NUM_FEATURES = 256

# 128 Pakete passen zu fast allen Experiment-2-Dateien.
WINDOW_SIZE = 128
STEP_SIZE = 64

# Experiment 3 besitzt 500 Pakete pro Datei und würde sonst
# erheblich mehr Fenster erzeugen als Experiment 2.
# Deshalb werden pro Aufnahme höchstens zwei Fenster genutzt.
MAX_WINDOWS_PER_FILE = 2


# ============================================================
# DATENAUFTEILUNG
# ============================================================

# Anteil der vorhandenen Trainingsdateien, der als neue
# Validation verwendet wird.
VALIDATION_FRACTION = 0.20

RANDOM_SEED = 42


# ============================================================
# SKALIERUNG
# ============================================================

# Gemeinsame Standardisierung beider Experimente.
#
# Der Scaler wird später ausschließlich auf den kombinierten
# Trainingsdaten angepasst und anschließend unverändert auf
# Validation und Test angewendet.
SCALER_TYPE = "standard"

SCALER_PATH = MODEL_DIR / "combined_standard_scaler.joblib"


# ============================================================
# CNN-TRAINING
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005

EARLY_STOPPING_PATIENCE = 12
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

MODEL_PATH = MODEL_DIR / "best_csi_cnn.keras"
HISTORY_PATH = RESULTS_DIR / "cnn_training_history.csv"