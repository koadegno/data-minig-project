mport random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from time import perf_counter


def get_data():
    chunks = []
    chuncks_folder = Path("chucks")

    print("Gathering dataset")
    for chunck in chuncks_folder.iterdir():
        print(chunck)
        chunks.append(
            pd.read_csv(chunck, sep=";", parse_dates=True, index_col="timestamps_UTC")
        )
        break

    data = pd.concat(chunks)
    print(data.head())
    return data


# plotting

print("running PLot")


def plot_clustering_results(
    features_list,
    X_train,
    result,
    png_filename="outlier_detection_plot_isolation_forest",
):
    num_columns = X_train.shape[1]
    fig, axes = plt.subplots(num_columns, num_columns, figsize=(30, 30))
    for i in range(num_columns):
        for j in range(num_columns):
            ax = axes[i, j]
            if i == j:
                ax.hist(X_train[:, i], bins=50, color="skyblue", alpha=0.7)
                ax.set_xlabel(f"Feature {features_list[i]}")
                ax.set_ylabel("Frequency")
            else:
                ax.scatter(X_train[:, j], X_train[:, i], s=3, color="blue", alpha=0.5)
                outliers_i = X_train[result == -1][:, i]
                outliers_j = X_train[result == -1][:, j]
                ax.scatter(outliers_j, outliers_i, s=20, color="red", alpha=0.5)
                ax.set_xlabel(f"Feature {features_list[j]}")
                ax.set_ylabel(f"Feature {features_list[i]}")
    fig.legend(["Inliers", "Outliers"], loc="upper right")

    for i in range(num_columns):
        axes[num_columns - 1, i].set_xlabel(f"Feature {features_list[i]}")
        axes[i, 0].set_ylabel(f"Feature {features_list[i]}")

    plt.tight_layout()
    plt.savefig(png_filename)
    print("Plot done!")
    # plt.show()


# Add the outlier predictions to the original DataFrame
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


def perform_grid_search(df: pd.DataFrame, X_train, clf, param_grid, num_iterations=10):
    num_iterations = 10
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)
    for i in range(num_iterations):
        # Choix aléatoire des paramètres
        params = {param: random.choice(values) for param, values in param_grid.items()}
        result_filename = (
            str(params)
            .replace("{", "")
            .replace("}", "")
            .replace(":", "-")
            .replace(",", "_")
            .replace(" ", "")
            .strip()
        )
        result_filename = (
            f"cluster_{i}_outlier_detection_plot_isolation_forest_"
            + result_filename
            + ".png"
        )
        result_filename_img = result_filename + ".png"
        result_filename_csv = result_filename + ".csv"

        result_filename_img = results_folder / result_filename_img
        result_filename_csv = results_folder / result_filename_csv
        # Création du modèle avec les paramètres choisis
        clf = IsolationForest(**params, n_jobs=-1)

        # Fit du modèle et prédiction sur les données d'entraînement
        result = clf.fit_predict(X_train)

        plot_clustering_results(features_list, X_train, result, result_filename_img)

        df[f"cluster_{i}"] = result
        cluster_means = df.groupby(f"cluster_{i}").mean()
        print(f"Cluster Means for {params}:")
        print(cluster_means.head())
        cluster_means.to_csv(result_filename_csv)

        count_feature_usage(
            clf, results_folder / Path(f"cluster_{i}_most_used_features.csv")
        )
    return df


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
    name = "pie_plot_most_used_features_" + str(filename.name).split(".")[0] + ".png"
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
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "contamination": [0.01, 0.05, 0.1, 0.2],
        "max_samples": [50, 100, 200, "auto"],
        "max_features": [2, 4, 6, len(features_list)],
    }
    df2 = data.copy()
    df2 = perform_grid_search(
        df2,
        X_train_normalized,
        clf,
        param_grid,
    )
    print("Majority Vote")
    temp = df2.drop(columns=features_list)
    temp["final_cluster"] = temp.mode(axis=1)[0]
    df2["final_cluster"] = temp["final_cluster"]
    cluster_means = df2.groupby("final_cluster").mean()
    print(f"Final Cluster Means:")
    print(cluster_means.head())
    cluster_means.to_csv("final_cluster_csv")

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
