import argparse
import os
from collections import defaultdict
from tbparse import SummaryReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# openai style plots
sns.set(style="darkgrid")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True,
        help="path to directory containing all the experiments")
    parser.add_argument("--algos", type=str, nargs="+", required=True,
        help="algorithm to generate plot for")
    parser.add_argument("--envs", type=str, nargs="+", required=True,
        help="environmets for which to present the analysis")
    args = parser.parse_args()
    return args

def process_file(log_path, algo_name="some_algo", is_spectral=False):
    data = SummaryReader(log_path, pivot=True).scalars
    data["return_mean"] = data["return_mean"].rolling(10,min_periods=1).mean()
    regularization = "spectral" if is_spectral else "normal"
    assert len(data["return_mean"]) == len(data["step"]), "bad file"
    res = {"return_mean": data["return_mean"], "step":data["step"], "algorithm": [algo_name]*len(data["step"]), "regularization": [regularization]*len(data["step"])}
    return pd.DataFrame(res)

def get_algos_files(args):
    dirs = [f for f in os.listdir(args.logdir) if os.path.isdir(os.path.join(args.logdir, f))]
    dirs = sorted(dirs)
    envs_dict = defaultdict(list)
    for d in dirs:
        for env in args.envs:
            if env in d:
                is_spectral = "spectral" in d
                algo = None
                for a in args.algos:
                    if a in d:
                        algo = a
                        break
                if algo is None:
                    continue
                processed_data = process_file(os.path.join(args.logdir, d), algo, is_spectral)
                if env not in envs_dict:
                    envs_dict[env]=[processed_data]
                else:
                    envs_dict[env].append(processed_data)
    return envs_dict

if __name__=="__main__":
    args = parse_args()
    envs_dict = get_algos_files(args)
    fig, axes = plt.subplots(1, len(envs_dict), figsize=(len(envs_dict)*5, 4))
    if len(envs_dict)==1: axes = [axes]

    for idx, env in enumerate(envs_dict):
        data = pd.concat(envs_dict[env])
        data.reset_index(inplace=True)
        sns.lineplot(ax=axes[idx], data=data, x="step", y="return_mean", hue="algorithm", style="regularization").set(title=env)
    
    plt.tight_layout()
    plt.savefig("test.pdf")
    plt.show()
