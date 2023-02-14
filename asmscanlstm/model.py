import json
import os
import pickle
from importlib.resources import files

import numpy as np
import tensorflow as tf
from progress.bar import Bar
from progress.spinner import Spinner


class ASMscanLSTM:

    def __init__(self) -> None:
        self.models = self._load_models()
        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()

    def predict(self, sequences: any) -> tuple[list[float], list[str]]:
        '''
            Generates output predictions for the input amino acids sequences.

            Parameters
            ----------
            sequences : array-like of shape (n_samples,)
                Amino acids sequences.

            Returns
            -------
            predictions : list[float] of shape (n_samples)
                List of probabilities of assigning a positive class.

            fragments : list[str] of shape (n_samples, 40)
                List of the most significant fragments of the sequences.
        '''
        T = self.config["T"]

        # Fragment protein sequences
        frags, scopes = self._fragment_sequences(sequences, T)
      
        # Tokenize text
        data = self.tokenizer.texts_to_sequences(frags)

        # Pad sequences
        data = tf.keras.preprocessing.sequence.pad_sequences(data, T)

        # Predict
        pred = []
        with Bar("Predicting", max=len(self.models), suffix="%(percent)d%%") as bar:
            for model in self.models:
                pred.append(model(data).numpy().flatten())
                bar.next()
        pred = np.mean(pred, axis=0)

        return self._to_sequence_prediction(frags, pred, scopes)
    
    def _fragment_sequences(self, sequences: any, max_seq_len: int) -> tuple[list[str], list[int]]:
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
    
    def _to_sequence_prediction(self, fragments: list[str], fragments_prediction: np.ndarray[np.float32], scopes: list[int]) -> tuple[list[float], list[str]]:
        pred = []
        frags = []

        p = 0
        for ss in scopes:
            scoped_frags_pred = fragments_prediction[p:p+ss]
            max_pred_index = np.argmax(scoped_frags_pred)
            pred.append(scoped_frags_pred[max_pred_index])
            frags.append(fragments[p+max_pred_index])
            p += ss

        return pred, frags
    
    def _load_tokenizer(self) -> tf.keras.preprocessing.text.Tokenizer:
        return pickle.load(files("asmscanlstm.resources").joinpath("tokenizer.pickle").open("rb"))

    def _load_models(self) -> list[tf.keras.models.Model]:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        models = []

        models_filepaths = list(files("asmscanlstm.resources.model.cvms").iterdir())
        spinner = Spinner("Loading model ")
        isFinished = True
        i = 0

        while isFinished:
            models.append(tf.keras.models.load_model(models_filepaths[i]))
            i += 1
            if i >= len(models_filepaths):
                isFinished = False
            spinner.next()

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        return models
    
    def _load_config(self) -> dict:
        return json.load(files("asmscanlstm.resources.model").joinpath("config.json").open("r"))
