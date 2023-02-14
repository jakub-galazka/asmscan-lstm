import glob
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from evaluate import evaluate
from util.data_loader import read_fasta
from util.file_explorer import (CONFIG_FILENAME, CV_MODELS_DIR, DATA_PRED_DIR,
                                MODELS_DIR, NEG_DATA_DIR, POS_DATA_DIR, SEP,
                                makedir)
from util.json import from_json, to_json
from util.tokenizer import load_tokenizer


TST_SETS_FILEPATHS = [
    os.path.join(NEG_DATA_DIR, "DisProt", "DisProt_test.fa"),
    os.path.join(NEG_DATA_DIR, "NLReff", "NLReff_test.fa"),
    os.path.join(NEG_DATA_DIR, "PB40", "PB40_1z20_clu50_test.fa"),
    os.path.join(NEG_DATA_DIR, "PB40", "PB40_1z20_clu50_test_sampled10000.fa"),
    os.path.join(POS_DATA_DIR, "bass_domain", "bass_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "bass_domain", "bass_other_ctm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "bass_domain", "bass_other_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "bass_motif", "bass_ntm_motif_test.fa"),
    os.path.join(POS_DATA_DIR, "bass_motif", "bass_ntm_motif_env5_test.fa"),
    os.path.join(POS_DATA_DIR, "bass_motif", "bass_ntm_motif_env10_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_domain", "fass_ctm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_domain", "fass_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_domain", "het-s_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_domain", "pp_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_domain", "sigma_ntm_domain_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ctm_motif_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ctm_motif_env5_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ctm_motif_env10_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ntm_motif_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ntm_motif_env5_test.fa"),
    os.path.join(POS_DATA_DIR, "fass_motif", "fass_ntm_motif_env10_test.fa")
]

class FragmentedSet:

    def __init__(self, fasta_filepath: str, max_seq_len: int) -> None:
        filepath_comps = fasta_filepath.split(os.sep)
        self.set_name = filepath_comps[-1].split(".")[0]
        self.isPositive = filepath_comps[1] == POS_DATA_DIR.split(os.sep)[1]

        self.ids, seqs = read_fasta(fasta_filepath)
        self.frags, self.scopes = self.fragment_sequences(seqs, max_seq_len)

    def fragment_sequences(self, sequences: list[str], max_seq_len: int) -> tuple[list[str], list[int]]:
        frags = []
        scopes = []

        for seq in sequences:
            seq_len = len(seq)

            if seq_len > max_seq_len:
                frags_number = seq_len - max_seq_len + 1

                for i in range(frags_number):
                    frags.append(seq[i:i+max_seq_len])

                scopes.append(frags_number)
            else:
                frags.append(seq)
                scopes.append(1)

        return frags, scopes

def predict(model_dir: str) -> None:
    tokenizer = load_tokenizer()
    config_filepath = os.path.join(model_dir, CONFIG_FILENAME) 
    config = from_json(config_filepath)
    T = config["T"]

    # Save modelcomb name to config
    cv_models_filepaths = glob.glob(os.path.join(model_dir, CV_MODELS_DIR, "*"))
    config["modelcomb_name"] = config["model_name"] + "comb" + "".join(str(i) for i in range(1, len(cv_models_filepaths) + 1))
    to_json(config_filepath, config)

    for set_filepath in TST_SETS_FILEPATHS:
        # Fragment protein sequences
        fs = FragmentedSet(set_filepath, T)
      
        # Tokenize text
        x_tst = tokenizer.texts_to_sequences(fs.frags)

        # Pad sequences
        x_tst = tf.keras.preprocessing.sequence.pad_sequences(x_tst, T)

        y_pred = []
        for i, model_filepath in enumerate(cv_models_filepaths):
            # Load model
            model = tf.keras.models.load_model(model_filepath)

            # Predict
            y_pred.append(model(x_tst).numpy().flatten()) # [[1], [1], ..., [1]] -> [1, 1, ..., 1]

            # Save cv results
            model_name = os.path.basename(model_filepath)
            save_model_prediction(model_name, fs, y_pred[i])

        # Save comb results
        if i > 0:
            y_pred = np.mean(y_pred, axis=0)
        save_model_prediction(config["modelcomb_name"], fs, y_pred)

def save_model_prediction(model_name: str, fs: FragmentedSet, fragments_prediction: np.ndarray[np.float32]) -> None:
    pred, frags = to_sequence_prediction(fs, fragments_prediction)
    save_prediction(
        os.path.join(model_dir, DATA_PRED_DIR, f"{fs.set_name}.{model_name}.csv"),
        fs, pred, frags
    )

def to_sequence_prediction(fs: FragmentedSet, fragments_prediction: np.ndarray[np.float32]) -> tuple[list[float], list[str]]:
    pred = []
    frags = []

    p = 0
    for ss in fs.scopes:
        scoped_frags_pred = fragments_prediction[p:p+ss]
        max_pred_index = np.argmax(scoped_frags_pred)
        pred.append(scoped_frags_pred[max_pred_index])
        frags.append(fs.frags[p+max_pred_index])
        p += ss

    return pred, frags

def save_prediction(filepath: str, fs: FragmentedSet, prediction: list[float], fragments: list[str]) -> None:
    df = pd.DataFrame({
        "id": fs.ids,
        "prob": prediction,
        "class": np.full(len(fs.ids), int(fs.isPositive)),
        "frag": fragments
    })
    df.to_csv(makedir(filepath), sep=SEP, index=False)

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, sys.argv[1])
    predict(model_dir)
    evaluate(model_dir)
    