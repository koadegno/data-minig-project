from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import plot_functions as plt_fct

debug = False

if debug:
    chunks_reader = pd.read_csv(
        "ar41_with_k_algo.csv", sep=",", parse_dates=True, index_col="timestamps_UTC", chunksize=10
    )

    for chunk in chunks_reader:
        df = chunk
        break
else:
    print("Reading csv ...")
    df = pd.read_csv("ar41_with_k_algo.csv", sep=",", parse_dates=True, index_col="timestamps_UTC")


result_type = df["cluster"].unique().tolist()
print("cluster type :", result_type)
result = df["cluster"].to_numpy()
print("cluster: ", result)
features_list = [
    "RS_E_InAirTemp_PC1",
    "RS_E_InAirTemp_PC2",
    "RS_E_OilPress_PC1",
    "RS_E_OilPress_PC2",
    "RS_E_RPM_PC1",
    "RS_E_RPM_PC2",
    "RS_E_WatTemp_PC1",
    "RS_E_WatTemp_PC2",
    "RS_T_OilTemp_PC1",
    "RS_T_OilTemp_PC2",
    "temperature",
    "precipitation",
    "windspeed_10m",
    "sum_pollen",
]
results_folder = Path("result_k_algo")
print("Plot save to: ", results_folder)
result_filename = "k_algo"
data_values = df[features_list].to_numpy()
scaler = RobustScaler()
data_values = scaler.fit_transform(data_values)

print("Plotting ...")
plt_fct.plot_feature_distribution(
    results_folder,
    result_filename,
    data_values,
    result,
    features_list,
    result_type,
)
