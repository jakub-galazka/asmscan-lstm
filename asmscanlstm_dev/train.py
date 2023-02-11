import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from history import history
from util.data_loader import load_data
from util.file_explorer import (CONFIG_FILENAME, CV_MODELS_DIR, DATA_HIST_DIR,
                                MODELS_DIR, NEG_DATA_DIR, POS_DATA_DIR,
                                makedir)
from util.json import to_json
from util.preprocessing import pad_protein_sequences
from util.tokenizer import load_tokenizer


# Model params
MODEL_NAME = "bass-model"
CV_ITERS = 2
D = 8
M = 8
DROPOUT = .1
EPOCHS = 1

def train(model_dir: str, cv_iters: int) -> None:
    tokenizer = load_tokenizer()
    V = len(tokenizer.word_index) + 1

    model = None
    for cv_i in range(1, cv_iters + 1):
        # Load data
        x_trn, y_trn = load_data(
            os.path.join(NEG_DATA_DIR, "PB40", f"PB40_1z20_clu50_trn{cv_i}.fa"),
            os.path.join(POS_DATA_DIR, "bass_motif", f"bass_ctm_motif_trn{cv_i}.fa")
        )
        x_val, y_val = load_data(
            os.path.join(NEG_DATA_DIR, "PB40", f"PB40_1z20_clu50_val{cv_i}.fa"),
            os.path.join(POS_DATA_DIR, "bass_motif", f"bass_ctm_motif_val{cv_i}.fa")
        )

        # Pad protein sequences
        T = len(max(x_trn, key=len))
        x_trn = pad_protein_sequences(x_trn, T)
        x_val = pad_protein_sequences(x_val, T)

        # Tokenize text
        x_trn = np.asarray(tokenizer.texts_to_sequences(x_trn))
        x_val = np.asarray(tokenizer.texts_to_sequences(x_val))

        # Build model
        i = tf.keras.layers.Input(shape=(T,), name="input")
        x = tf.keras.layers.Embedding(V, D, name="embedding")(i)
        x = tf.keras.layers.Dropout(DROPOUT, name="dropout_0")(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(M, return_sequences=True), name="bi-lstm")(x) # Instruction at the bottom
        x = tf.keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
        x = tf.keras.layers.LSTM(int(M / 2), name="lstm")(x)
        x = tf.keras.layers.Dropout(DROPOUT, name="before-classif")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="classif")(x)
        model = tf.keras.models.Model(i, x, name=MODEL_NAME)

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=[
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                tf.keras.metrics.SensitivityAtSpecificity(.99, name="sens_at_spec_99"),
                tf.keras.metrics.SensitivityAtSpecificity(.999, name="sens_at_spec_999")
            ]
        )

        # Train model
        print(f"\n_________________________ CV Iteration {cv_i} / {cv_iters} _________________________\n")
        r = model.fit(
            x_trn,
            y_trn,
            epochs=EPOCHS,
            validation_data=(x_val, y_val)
        )

        # Save results
        model_name = MODEL_NAME + str(cv_i)
        model.save(makedir(os.path.join(model_dir, CV_MODELS_DIR, model_name)))
        np.save(makedir(os.path.join(model_dir, DATA_HIST_DIR, model_name)), r.history)

    # Save architecture
    with open(os.path.join(model_dir, "architecture.txt"), "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    to_json(os.path.join(model_dir, CONFIG_FILENAME), {"model_name": MODEL_NAME, "T": T})

def test_mode() -> bool:
    if ("--test" in sys.argv) or ("-t" in sys.argv):
        seed = 1
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        return True
    return False

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, str(time.time()))

    cv_iters = CV_ITERS
    if test_mode():
        cv_iters = 1
        model_dir += "-test"

    train(model_dir, cv_iters)
    history(model_dir)



'''
    Bidirectional (https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)

    return_sequences=True -> (N,T,D) == (samples, time_steps, hidden_state_dim)
    return_sequences=False -> return_sequences[:,-1] (last_time_step)

    Bidirectional wrapper when return_sequences=False -> fun([forward_output, backward_output]) -> r = fun([(1..n), (n..1)]) -> output: r[-1] == fun([forward_n_time_step, backward_1_time_step])
    Bidirectional wrapper when return_sequences=True -> fun([forward_output, reverse(backward_output)]) -> output: r = fun([(1..n), (1..n)]) -> r[-1] == fun([forward_n_time_step, backward_n_time_step])
        
    *fun = {sum, mul, concat (default), ave, None}
'''
