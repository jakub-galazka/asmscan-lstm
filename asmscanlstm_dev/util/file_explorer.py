import os


PACKAGE_ROOT = "asmscanlstm"

# Data dirs
NEG_DATA_DIR = os.path.join(PACKAGE_ROOT, "resources", "data", "negative")
POS_DATA_DIR = os.path.join(PACKAGE_ROOT, "resources", "data", "positive")

# Save dirs
MODELS_DIR = os.path.join(PACKAGE_ROOT, "models")
CV_MODELS_DIR = "cvms"
CONFIG_FILENAME = "config.json"
DATA_HIST_DIR = os.path.join("data", "history")
DATA_PRED_DIR = os.path.join("data", "predictions")

SEP = "\t"

def makedir(dirpath: str) -> str:
    if len(os.path.split(dirpath)) > 1:
        directory = os.path.dirname(dirpath)
    else:
        directory = dirpath
    os.makedirs(directory, exist_ok=True)
    return dirpath
