import os
import pickle

import tensorflow as tf

from .data_loader import load_data
from .file_explorer import NEG_DATA_DIR, PACKAGE_ROOT, POS_DATA_DIR


TOKENIZER_PATH = os.path.join(PACKAGE_ROOT, "resources", "tokenizer.pickle")

def load_tokenizer() -> tf.keras.preprocessing.text.Tokenizer:
    if not os.path.isfile(TOKENIZER_PATH):
        create_tokenizer()
    with open(TOKENIZER_PATH, "rb") as handle:
        return pickle.load(handle)

def create_tokenizer() -> None:
    seqs, _ = load_data(
        os.path.join(NEG_DATA_DIR, "PB40", "PB40_1z20_clu50_trn123456.fa"),
        os.path.join(POS_DATA_DIR, "bass_motif", "bass_ctm_motif_trn123456.fa")
    )

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(seqs)

    with open (TOKENIZER_PATH, "wb") as handle:
        pickle.dump(tokenizer, handle, pickle.HIGHEST_PROTOCOL)
