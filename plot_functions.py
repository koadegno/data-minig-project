from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


def plot_feature_distribution(
    results_folder: Path,
    result_filename: str,
    data_values: np.ndarray,
    result: np.ndarray,
    features_list: List,
    result_type: List,
):
    """plot distribution of each feature

    Args:
        results_folder (Path): The folder to save the plots in
        result_filename (str): the suffix of the result filename
        data_values (np.ndarray): the data in form of np array
        result (np.ndarray): The result of the classification
        features_list (List): The list of the name of each feature
    """
    basic_colors = [
        "b",  # blue
        "g",  # green
        "r",  # red
        "c",  # cyan
        "m",  # magenta
        "y",  # yellow
        "k",  # black
        "w",  # white
    ]
    results_folder.mkdir(exist_ok=True)
    results_folder = results_folder / "features"
    results_folder.mkdir(exist_ok=True)
    cluster_data_values = {}

    for cluster_type in result_type:
        cluster_data_values[cluster_type] = data_values[result == cluster_type]

    for i, feature in tqdm(enumerate(features_list)):
        plt.figure(figsize=(8, 6))

        cluster_min_list = []
        cluster_max_list = []
        cluster_mean = [0 for _ in range(len(cluster_data_values))]
        cluster_median = [0 for _ in range(len(cluster_data_values))]
        for cluster_type, cluster_list in cluster_data_values.items():
            # compute bins
            min_value = cluster_list[:, i].min()
            max_value = cluster_list[:, i].max()
            cluster_min_list.append(min_value)
            cluster_max_list.append(max_value)
            bins = np.linspace(min(cluster_min_list), max(cluster_max_list), num=50)

            # compute mean and median
            mean = np.mean(cluster_list[:, i])
            median = np.median(cluster_list[:, i])
            cluster_mean[cluster_type] = mean
            cluster_median[cluster_type] = median

        for cluster_type, cluster_list in cluster_data_values.items():
            plt.hist(
                cluster_list[:, i],
                bins=bins,
                color=basic_colors[cluster_type],
                alpha=0.7,
                label=f"cluster {cluster_type}",
            )

            plt.axvline(
                cluster_mean[cluster_type],
                color=basic_colors[cluster_type],
                linestyle="dashed",
                linewidth=2,
                label=f"cluster {cluster_type} Mean: {cluster_mean[cluster_type]:.2f}",
            )

            plt.axvline(
                cluster_median[cluster_type],
                color=basic_colors[cluster_type],
                linestyle="dotted",
                linewidth=2,
                label=f"cluster {cluster_type} Median: {cluster_median[cluster_type]:.2f}",
            )

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature} for Clusters")
        plt.legend()
        # plt.xlim(inliers_list[:, i].min(), outliers_list[:, i].max())  # Adjust the x-axis limits
        plt.savefig(results_folder / (result_filename + f"_Distribution of {feature} for Cluster.png"))
