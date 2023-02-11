import glob
import os
import sys

import pandas as pd

from util.file_explorer import (CONFIG_FILENAME, DATA_PRED_DIR, MODELS_DIR,
                                SEP, makedir)
from util.json import from_json
from util.metricer import compute_metrics


PRED_COMBS = [
    ["PB40_1z20_clu50_test", "bass_ntm_motif_test"],
    ["PB40_1z20_clu50_test", "bass_ntm_motif_env5_test"],
    ["PB40_1z20_clu50_test", "bass_ntm_motif_env10_test"],
    ["PB40_1z20_clu50_test", "fass_ctm_motif_test"],
    ["PB40_1z20_clu50_test", "fass_ctm_motif_env5_test"],
    ["PB40_1z20_clu50_test", "fass_ctm_motif_env10_test"],
    ["PB40_1z20_clu50_test", "fass_ntm_motif_test"],
    ["PB40_1z20_clu50_test", "fass_ntm_motif_env5_test"],
    ["PB40_1z20_clu50_test", "fass_ntm_motif_env10_test"],
    ["PB40_1z20_clu50_test", "bass_ntm_domain_test"],
    ["PB40_1z20_clu50_test", "fass_ctm_domain_test"],
    ["PB40_1z20_clu50_test", "fass_ntm_domain_test"],
    ["NLReff_test", "bass_ntm_domain_test"],
    ["NLReff_test", "bass_other_ctm_domain_test", "bass_other_ntm_domain_test"],
    ["NLReff_test", "fass_ntm_domain_test"],
    ["NLReff_test", "fass_ctm_domain_test"],
    ["NLReff_test", "fass_ntm_domain_test"],
    ["NLReff_test", "het-s_ntm_domain_test", "pp_ntm_domain_test", "sigma_ntm_domain_test"],
]

COLUMNS = ["Model", "#pos", "#neg", "AUROC", "AP"]
RC_FPR_LABEL = "Rc|FPR1e-"

class ModelPredComb:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.y_true = []
        self.y_pred = []
        self.neg = 0
        self.pos = 0

    def get_metrics(self) -> list:
        roc_auc, pr_auc, self.tpr = compute_metrics(self.y_true, self.y_pred)

        metrics = [
            self.model_name,
            self.pos,
            self.neg,
            roc_auc,
            pr_auc
        ]
        metrics.extend(self.tpr)

        return metrics

    def get_columns(self) -> list[str]:
        return COLUMNS + [RC_FPR_LABEL + str(i) for i in range(1, len(self.tpr) + 1)] # Generate Rc|FPR columns

    def add(self, pred_csv: pd.DataFrame) -> None:
        y_true = pred_csv["class"]
        self.y_true.extend(y_true)
        self.y_pred.extend(pred_csv["prob"])

        n = pred_csv.shape[0]
        if y_true[0]:
            self.pos += n
        else:
            self.neg += n

def evaluate(model_dir: str) -> None:
    config = from_json(os.path.join(model_dir, CONFIG_FILENAME))

    # Group predictions (based on combinations)
    for comb in PRED_COMBS:
        models_pred_combs: dict[str, ModelPredComb] = {}

        # Group predictions (based on cv models)
        for set in comb:
            for pred_filepath in glob.glob(os.path.join(model_dir, DATA_PRED_DIR, f"{set}*")):
                csv = pd.read_csv(pred_filepath, sep=SEP)
                model_name = os.path.basename(pred_filepath).split(".")[1]

                if not model_name in models_pred_combs.keys():
                    models_pred_combs[model_name] = ModelPredComb(model_name)
                models_pred_combs[model_name].add(csv)

        # Calculate metrics
        data = [model_pc.get_metrics() for model_pc in models_pred_combs.values()]
        columns = models_pred_combs[f"{model_name}"].get_columns()
        df = pd.DataFrame(data, columns=columns)

        # Calculate AVG
        if df.shape[0] > 1:
            cvs = df.iloc[:-1]
            cvs_avg = cvs.mean(numeric_only=True)
            cvs_avg[columns[0]] = config["model_name"] + "-avg"
            df.loc[len(df) - 1.5] = cvs_avg
            df = df.sort_index()

        # Save results
        df.to_csv(makedir(os.path.join(model_dir, "data", "summaries", ".".join(comb) + ".csv")), sep=SEP, index=False)

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, sys.argv[1])
    evaluate(model_dir)
