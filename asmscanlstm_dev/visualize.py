import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cycler import cycler
from util.file_explorer import (CONFIG_FILENAME, CV_MODELS_DIR, DATA_PRED_DIR,
                                MODELS_DIR, SEP)
from util.json import from_json
from util.ploter import savefig
from util.tokenizer import load_tokenizer
from util.tsne import tsne_2d

TSNE_COMBS = [
    ["PB40_1z20_clu50_test_sampled100", "NLReff_test_sampled100", "fass_ctm_domain_test", "fass_ntm_domain_test", "bass_ntm_domain_test"],
    ["PB40_1z20_clu50_test_sampled100", "NLReff_test_sampled100", "fass_ctm_motif_test", "fass_ntm_motif_test", "bass_ntm_motif_test"],
    ["PB40_1z20_clu50_test_sampled100", "NLReff_test_sampled100", "bass_ntm_motif_test", "bass_ntm_motif_env5_test", "bass_ntm_motif_env10_test"],
    ["PB40_1z20_clu50_test_sampled100", "NLReff_test_sampled100", "fass_ntm_motif_test", "fass_ntm_motif_env5_test", "fass_ntm_motif_env10_test"],
    ["PB40_1z20_clu50_test_sampled100", "NLReff_test_sampled100", "fass_ctm_motif_test", "fass_ctm_motif_env5_test", "fass_ctm_motif_env10_test"]
]

MARKER_SIZE = mpl.rcParams["lines.markersize"] ** 2 # Default marker size (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)

def visualize(model_dir: str, layer_name: str) -> None:
    tokenizer = load_tokenizer()
    config = from_json(os.path.join(model_dir, CONFIG_FILENAME))

    for comb in TSNE_COMBS:
        # Collect the most significant fragments (comb results)
        sets_sizes = {}
        frags = []
        marker_sizes = []
        for set in comb:
            pred = pd.read_csv(os.path.join(model_dir, DATA_PRED_DIR, f'{set}.{config["modelcomb_name"]}.csv'), sep=SEP)
            p = pred["prob"]
            c = pred["class"]
            f = pred["frag"]

            d = abs(c - p)

            sets_sizes[set] = len(f)
            frags.extend(f)
            marker_sizes.extend(d)
        marker_sizes = 1.2 * MARKER_SIZE - np.multiply(marker_sizes, MARKER_SIZE)

        # Tokenize text
        frags = tokenizer.texts_to_sequences(frags)

        # Pad sequences
        frags = tf.keras.preprocessing.sequence.pad_sequences(frags, config["T"])

        # Collect multidimensional representations from cv models
        mdim_rep = []
        for model_filepath in glob.glob(os.path.join(model_dir, CV_MODELS_DIR, "*")):
            # Load model
            model = tf.keras.models.load_model(model_filepath)

            # Get output from selected layer
            layer_out = model.get_layer(layer_name).output
            fun = tf.keras.backend.function(model.input, layer_out)
            mdim_rep.append(fun(frags))

        # T-distributed Stochastic Neighbor Embedding
        mdim_rep = np.concatenate(mdim_rep, axis=1)
        x, y = tsne_2d(mdim_rep)

        # Config plot
        plt.figure(figsize=(12.8, 9.6))
        plt.rc("axes", prop_cycle=cycler(color=plt.cm.rainbow(np.linspace(1, 0, len(comb))))) # 0, 1: Violet -> Red | 1, 0: Red -> Violet
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot TSNE
        i = 0
        for set in comb:
            ss = sets_sizes[set]
            plt.scatter(x[i:i+ss], y[i:i+ss], label=set, s=marker_sizes[i:i+ss])
            i += ss
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.tight_layout()
        savefig(os.path.join(model_dir, "plots", "tsne", f'{".".join(comb)}.png'), legend=False)

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, sys.argv[1])
    layer_name = "before-classif" if len(sys.argv) < 3 else sys.argv[2] 
    visualize(model_dir, layer_name)
