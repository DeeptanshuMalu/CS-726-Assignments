import os
import glob
import numpy as np

STEPS_VALUES = [10, 50, 100, 150, 200, 500]

BETA_PAIRS = [
    (0.0001, 0.02),
    (0.001, 0.2),
    (0.00001, 0.002),
    (0.00001, 0.02),
    (0.0001, 0.2),
    (0.00001, 0.2),
]

DATASETS = [
    "moons",
    "circles",
    "blobs",
    "manycircles",
    "helix",
]

if __name__ == "__main__":
    with open("var_timesteps.txt", "w") as f:
        for step in STEPS_VALUES:
            ubeta = 0.02
            lbeta = 0.0001
            for dataset in DATASETS:
                if dataset == "helix":
                    dim = 3
                else:
                    dim = 2
                runname = f"exps/ddpm_{dim}_{step}_{lbeta}_{ubeta}_{dataset}"

                file_path = os.path.join(runname, "metrics.txt")

                with open(file_path, "r") as metrics_f:
                    metrics_content = metrics_f.read().strip()
                    f.write(f"{runname[5:]}:\n{metrics_content}\n")
                    f.write("\n")

    with open("var_beta.txt", "w") as f:
        for lbeta, ubeta in BETA_PAIRS:
            step = 200
            for dataset in DATASETS:
                if dataset == "helix":
                    dim = 3
                else:
                    dim = 2
                runname = f"exps/ddpm_{dim}_{step}_{lbeta}_{ubeta}_{dataset}"

                file_path = os.path.join(runname, "metrics.txt")

                with open(file_path, "r") as metrics_f:
                    metrics_content = metrics_f.read().strip()
                    f.write(f"{runname[5:]}:\n{metrics_content}\n")
                    f.write("\n")
