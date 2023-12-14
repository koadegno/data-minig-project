import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import plot_functions as plt_fct
import gc

debug = False
csv_data_filename = "ar41_with_k_algo.csv"

if debug:
    count = 1
    max_count = 3
    df = pd.DataFrame()
    chunks_reader = pd.read_csv(
        csv_data_filename, sep=",", parse_dates=True, index_col="timestamps_UTC", chunksize=300
    )

    for chunk in chunks_reader:
        if count == max_count:
            break
        else:
            df = pd.concat([df, chunk])
        count += 1

else:
    print(f"Reading csv ... {csv_data_filename}")
    df = pd.read_csv(csv_data_filename, sep=",", parse_dates=True, index_col="timestamps_UTC")


result_type = df["cluster"].unique().tolist()
print("cluster type :", result_type)
result = df["cluster"].to_numpy()
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
results_folder.mkdir(exist_ok=True)
print("Plot save to: ", results_folder)
result_filename = "k_algo"
data_values = df[features_list].to_numpy()
df = None
del df
gc.collect()

scaler = RobustScaler()
data_values = scaler.fit_transform(data_values)

png_filename = results_folder / "clustering_plots"
png_filename.mkdir(exist_ok=True)
png_filename = png_filename / f"cluster_detection_plot_{result_filename}.png"

print("Plotting ...")
plt_fct.plot_feature_distribution(
    results_folder=results_folder,
    result_filename=result_filename,
    data_values=data_values,
    result=result,
    features_list=features_list,
    result_type=result_type,
)
print("Plotting 2 ...")

plt_fct.plot_clustering_results(
    features_list,
    data_values,
    result,
    result_type,
    png_filename,
)
