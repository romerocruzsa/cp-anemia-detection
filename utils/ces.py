import numpy as np
import pandas as pd
import glob
import re

def load_metrics_csv(split, comp_method, mode):
    # to store all dfs from the csv files
    dfs = []

    models = ["mobilenetv2", "resnet18", "densenet121", "vgg16", "vit-tiny",
               "efficientnet-b0", "shufflenetv2-0.5x", "regnety-400mf",
               "mnasnet0_5", "convnext-tiny", "ghostnetv2", "tinynet-a"]

    precision = ["fp32","fp16","int8","int4"]

    # Load data from all CSVs
    for model in models:
        for bw in precision:
            # Load CSV in individual dataframe
            path = f"metrics/pytorch/{comp_method.lower()}/{split}_metrics_{model}_{comp_method.upper()}_{mode.upper()}.csv"
            csv_file = glob.glob(path)
            
            df = pd.read_csv(csv_file[0])

            
            # Append to the list of dataframes for all models
        dfs.append(df)

    # Concatenate all data into a single DataFrame
    dfs_all = pd.concat(dfs, ignore_index=True)
    
    return dfs_all

def extract_float(s):
    match = re.search(r'\d+\.\d+', s)
    return float(match.group()) if match else None

def CES(data):
    w_a = 0.9
    w_r = 0.6
    k = 5  # Sensitivity to accuracy degradation

    res_list = []

    for model in data["model"].unique():
        base = data[(data["model"] == model) & (data["bit-width"] == "fp32")].iloc[0]
        
        A_b = base["accuracy"]
        Sz_b = base["model_size"]
        L_b = base["latency"]
        
        for _, row in data[data["model"] == model].iterrows():
            A_c = row["accuracy"]
            Sz_c = row["model_size"]
            L_c = row["latency"]
            
            if A_b == 0 or Sz_b == 0 or L_b == 0 or Sz_c == 0 or L_c == 0:
                ces = None
            else:
                # Penalized accuracy term (centered at 0)
                acc_score = np.exp(-k * (1 - (A_c / A_b))) - 1

                # Resource gain (relative)
                size_gain = (Sz_b - Sz_c) / Sz_b
                latency_gain = (L_b - L_c) / L_b

                res = w_a * acc_score + w_r * (size_gain + latency_gain)
            
            res_list.append({
                "model": model,
                "bit-width": row["bit-width"],
                "accuracy": row["accuracy"],
                "model_size": row["model_size"],
                "latency": row["latency"],
                "CES": res
            })

    df = pd.DataFrame(res_list)
    return df
