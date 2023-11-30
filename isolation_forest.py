import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from time import perf_counter, time
import joblib
from sklearn.decomposition import PCA


def get_data(chunks_stop=None):
    chunks = []
    chunks_folder = Path("chucks")

    print("Gathering dataset")
    chunks_counter = 1
    if chunks_stop is None:
        chunks_stop = -1

    for chunk in chunks_folder.iterdir():
        print(chunk)
        chunks.append(
            pd.read_csv(chunk, sep=";", parse_dates=True, index_col="timestamps_UTC")
        )
        if chunks_counter == chunks_stop:
            break
        chunks_counter += 1

    data = pd.concat(chunks)
    print(data.head())
    return data


def plot_feature_distribution(
    results_folder: Path, result_filename, X_train, result, features_list
):
    outliers_train = X_train[result == -1]
    inliers_train = X_train[result == 1]
    results_folder = results_folder / "features"
    results_folder.mkdir(exist_ok=True)
    for i, feature in enumerate(features_list):
        plt.figure(figsize=(8, 6))
        plt.hist(
            inliers_train[:, i],
            bins=50,
            alpha=0.7,
            color="skyblue",
            label="Inliers",
        )
        plt.hist(
            outliers_train[:, i],
            bins=50,
            alpha=0.7,
            color="orange",
            label="Outliers",
        )
        inliers_mean = np.mean(inliers_train[:, i])
        inliers_median = np.median(inliers_train[:, i])
        outliers_mean = np.mean(outliers_train[:, i])
        outliers_median = np.median(outliers_train[:, i])

        plt.axvline(
            inliers_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Inliers Mean: {inliers_mean:.2f}",
        )
        plt.axvline(
            inliers_median,
            color="blue",
            linestyle="dotted",
            linewidth=2,
            label=f"Inliers Median: {inliers_median:.2f}",
        )
        plt.axvline(
            outliers_mean,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Outliers Mean: {outliers_mean:.2f}",
        )
        plt.axvline(
            outliers_median,
            color="red",
            linestyle="dotted",
            linewidth=2,
            label=f"Outliers Median: {outliers_median:.2f}",
        )
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature} for Inliers and Outliers")
        plt.legend()
        plt.xlim(
            inliers_train[:, i].min(), outliers_train[:, i].max()
        )  # Adjust the x-axis limits
        plt.savefig(
            results_folder
            / (
                result_filename
                + f"_Distribution of {feature} for Inliers and Outliers.png"
            )
        )


def plot_clustering_results(
    features_list,
    X_train,
    result,
    model,
    png_filename="outlier_detection_plot_isolation_forest",
):
    num_columns = X_train.shape[1]
    fig, axes = plt.subplots(num_columns, num_columns, figsize=(30, 30))
    for i in range(num_columns):
        for j in range(num_columns):
            ax = axes[i, j]
            if i == j:
                ax.hist(X_train[:, i], bins=50, color="skyblue", alpha=0.7)
                ax.set_xlabel(f"{features_list[i]}")
                ax.set_ylabel("Frequency")
            else:
                ax.scatter(X_train[:, j], X_train[:, i], s=3, color="blue", alpha=0.3)
                outliers_i = X_train[result == -1][:, i]
                outliers_j = X_train[result == -1][:, j]
                ax.scatter(outliers_j, outliers_i, s=3, color="orange", alpha=0.5)
                ax.set_xlabel(f"{features_list[j]}")
                ax.set_ylabel(f"{features_list[i]}")
    fig.legend(["", "Inliers", "Outliers"], loc="upper right")

    for i in range(num_columns):
        axes[num_columns - 1, i].set_xlabel(f"Feature {features_list[i]}")
        axes[i, 0].set_ylabel(f"{features_list[i]}")

    plt.tight_layout()
    plt.savefig(png_filename)

    print("Plot done!")
    # plt.show()


def generate_2d_plot(X_train, result):
    outliers_train = X_train[result == -1]
    inliers_train = X_train[result == 1]

    # Scatter plot for inliers in X_train
    plt.scatter(inliers_train[:, 0], inliers_train[:, 1], label="Inliers")

    # Scatter plot for outliers in X_train
    plt.scatter(
        outliers_train[:, 0], outliers_train[:, 1], label="Outliers", color="red"
    )

    plt.xlabel("RS_E_InAirTemp_PC1")
    plt.ylabel("RS_E_InAirTemp_PC2")
    plt.legend()
    plt.title("Outlier Detection within X_train")
    plt.show()


def perform_grid_search(df: pd.DataFrame, X_train, param_grid, num_iterations=10):
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)
    for i in range(num_iterations):
        # Choix aléatoire des paramètres
        params = {param: random.choice(values) for param, values in param_grid.items()}
        print(f"cluster {i} with parameters : {params}")
        result_filename = (
            str(params)
            .replace("{", "")
            .replace("}", "")
            .replace(":", "-")
            .replace(",", "_")
            .replace("'", "")
            .replace("-", "_")
            .replace(" ", "")
            .strip()
        )
        result_filename = "isolation_forest_" + result_filename
        result_filename_img = f"cluster_{i}_plot_" + result_filename + ".png"
        result_filename_csv = f"cluster_{i}_plot_" + result_filename + ".csv"

        result_filename_img = results_folder / result_filename_img
        result_filename_csv = results_folder / result_filename_csv

        # Création du modèle avec les paramètres choisis
        clf = IsolationForest(**params, n_jobs=-1, max_samples="auto")

        start_time = time()
        result = clf.fit_predict(X_train)
        end_time = time()
        joblib.dump(clf, results_folder / (result_filename + f"{i}_model.pkl"))
        # clf = joblib.load(results_folder / (result_filename + f"{i}_model.pkl"))
        print(f"Prediction time : {round(end_time - start_time, 2)} s")

        anomaly_scores = clf.decision_function(X_train)

        plot_feature_distribution(
            results_folder, result_filename, X_train, result, features_list
        )
        print("Done plotting feature distibution")

        plot_anomaly(results_folder, result_filename, anomaly_scores)
        print("Done ploting anomaly")

        plot_clustering_results(
            features_list, X_train, result, clf, result_filename_img
        )
        print("Done ploting result")

        df[f"cluster_{i}"] = result
        df[f"cluster_{i}"] = df[f"cluster_{i}"].astype("category")

        if num_iterations == 1:
            df.to_csv(
                results_folder / f"cluster_{i}_ar41_with_isolation_forest_cluster.csv"
            )
            print("Done saving clustered csv")

        cluster_means = df.groupby(f"cluster_{i}").mean()
        cluster_means.to_csv(result_filename_csv)

        count_feature_usage(
            clf, results_folder / Path(f"cluster_{i}_most_used_features.csv")
        )
        print("Done ploting count feature")

    return df


def plot_anomaly(results_folder, result_filename, anomaly_scores):
    plt.figure(figsize=(8, 6))
    plt.hist(anomaly_scores, bins="auto", density=True)
    plt.xlabel("Anomaly Scores")
    plt.ylabel("Density")
    plt.title("Distribution of Anomaly Scores (Decision Function)")
    plt.savefig(
        results_folder / (result_filename + "_Distribution of Anomaly Scores.png")
    )

    anomaly_df = pd.DataFrame({"Anomaly_Scores": anomaly_scores}, index=data.index)

    # Scatter plot anomaly scores against timestamps (index)
    plt.figure(figsize=(10, 6))
    plt.scatter(anomaly_df.index, anomaly_df["Anomaly_Scores"], s=5, c="red", alpha=0.5)
    plt.xlabel("Timestamps")
    plt.ylabel("Anomaly Scores")
    plt.title("Anomaly Scores over Time")
    plt.savefig(results_folder / (result_filename + "_Anomaly Scores over Time.png"))


def count_feature_usage(clf, filename=Path("most_used_features.csv")):
    # Accéder aux arbres de la forêt
    trees = clf.estimators_

    # Initialiser un dictionnaire pour compter les occurrences des caractéristiques utilisées
    feature_usage_count = {}

    # Parcourir chaque arbre dans la forêt
    for tree in trees:
        # Obtenez le nœud racine de l'arbre
        tree_root = tree.tree_

        # Parcourir les arbres de manière récursive
        def traverse_tree(node_id):
            # Récupérer la caractéristique du nœud
            feature_idx = tree_root.feature[node_id]
            # Si une caractéristique est utilisée pour le split, incrémenter son compteur dans le dictionnaire
            if feature_idx != -2:  # -2 indique un nœud de feuille
                feature = features_list[feature_idx]
                if feature not in feature_usage_count:
                    feature_usage_count[feature] = 1
                else:
                    feature_usage_count[feature] += 1

                # Récupérer les sous-arbres gauche et droit
                left_child = tree_root.children_left[node_id]
                right_child = tree_root.children_right[node_id]

                # Continuer la traversée de l'arbre pour les enfants gauche et droit
                traverse_tree(left_child)
                traverse_tree(right_child)

        # Démarrer la traversée de l'arbre à partir du nœud racine (index 0)
        traverse_tree(0)

    most_used_features = sorted(
        feature_usage_count.items(), key=lambda x: x[1], reverse=True
    )
    most_used_features_df = pd.DataFrame(
        most_used_features, columns=["Features", "Count"]
    )
    most_used_features_df.to_csv(filename, index=False)
    plt.figure(figsize=(8, 8))
    plt.pie(
        most_used_features_df["Count"],
        labels=most_used_features_df["Features"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title("Usage Counts of Features")
    plt.tight_layout()

    # Save the pie plot as an image
    name = str(filename.name).split(".")[0] + "_pie_plot_most_used_features.png"
    plot_filename = filename.with_name(name)
    plt.savefig(plot_filename)

    # print("Caractéristiques les plus utilisées :\n")
    # for feature, count in most_used_features:
    #     print(f"{feature}: {count} utilisations")
    return feature_usage_count


if __name__ == "__main__":
    df = get_data()
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

    data = df[features_list]

    X = data.to_numpy()
    print("Running splitting")

    train_size = 1  # %
    train_split = int(len(X) * train_size)
    X_train = X[:train_split]
    X_test = X[train_split:]
    print("length of X: ", len(X))
    print("length of X_train: ", len(X_train))
    print("length of X_test: ", len(X_test))
    print(f"split on :{train_split}")

    # clustering model
    preprocessor = RobustScaler()
    X_train_normalized = preprocessor.fit_transform(X_train)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_normalized)
    param_grid = {
        "n_estimators": [500],
        # "contamination": [0.01, 0.05, 0.1, 0.2],
        "contamination": [0.05, 0.1],
    }
    df2 = data.copy()
    df2 = perform_grid_search(df2, X_train_pca, param_grid, num_iterations=1)

    # print("Majority Vote")
    # temp = df2.drop(columns=features_list)
    # temp["final_cluster"] = temp.mode(axis=1)[0]
    # df2["final_cluster"] = temp["final_cluster"]
    # cluster_means = df2.groupby("final_cluster").mean()
    # print("Final Cluster Means")
    # cluster_means.to_csv("results/final_cluster.csv")

    # df["Cluster"] = result

    # print("Data with cluster")
    # print(df.head())

    # p1 = perf_counter()
    # result = clf.fit_predict(X_train_normalized)
    # p2 = perf_counter()
    # print(f"prediction times {p2-p1}")
    # count_feature_usage(clf)

    # plot_clustering_results(features_list, X_train_normalized, result)

    # Utilisation de la fonction avec un modèle Isolation Forest entraîné

    # Afficher les caractéristiques les plus utilisées
