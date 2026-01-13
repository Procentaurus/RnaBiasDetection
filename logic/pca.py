import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .params import (ROOT_DIR,
                     TRESHHOLD)


def filter_low_expression_genes(counts, min_fraction=0.5):
    """
    Keep genes expressed >= cutoff in at least min_fraction of samples.
    """
    expressed = (counts >= TRESHHOLD).sum(axis=1)
    min_samples = int(min_fraction * counts.shape[1])
    return counts.loc[expressed >= min_samples]


def plot_scatter(samples, pc1, pc2, data_col, palette=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=samples,
        x="PC1",
        y="PC2",
        hue=data_col,
        palette=palette,
        s=40
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)

    plt.xlabel(f"PC1 ({pc1*100:.1f}%)")
    plt.ylabel(f"PC2 ({pc2*100:.1f}%)")
    plt.title(f"PCA of RNA-seq samples by {data_col}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "images" / "pca" / f"colored_by_{data_col}.png")
    plt.close()


def plot_boxplot(samples):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x="AgeGroup", y="PC1", data=samples)
    plt.title("PC1 by AgeGroup")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="AgeGroup", y="PC2", data=samples)
    plt.title("PC2 by AgeGroup")

    plt.tight_layout()
    plt.savefig(ROOT_DIR / "images" / "pca" / "boxplots_by_AgeGroup.png")
    plt.close()


def plot_correlation_heatmap(data, label):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )
    plt.title(f"{label} Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "images" / "correlation" / f"{label}_heatmap.png")
    plt.close()


if __name__ == "__main__":

    # Load metadata
    samples = pd.read_csv(
        ROOT_DIR / "data/samples.csv",
        sep=";",
        decimal=",",
        index_col=0
    )

    for col in ["Age", "lowLeuko", "mediumLeuko", "highLeuko"]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")

    # Load expression matrix
    counts = pd.read_csv(
        ROOT_DIR / "data/counts_normalized.csv",
        sep=";",
        decimal=",",
        index_col=0
    )

    # Match samples
    samples = samples[samples["id"].isin(counts.columns)]
    counts = counts[samples["id"]]

    # -------------------------------
    # Low-expression filtering
    # -------------------------------

    counts_filt = filter_low_expression_genes(
        counts,
        min_fraction=0.5
    )

    print(f"Genes before filtering: {counts.shape[0]}")
    print(f"Genes after filtering:  {counts_filt.shape[0]}")

    # -------------------------------
    # PCA
    # -------------------------------

    X = counts_filt.T
    X_centered = StandardScaler(with_std=False).fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_centered)

    samples["PC1"] = pcs[:, 0]
    samples["PC2"] = pcs[:, 1]

    pc1_var, pc2_var = pca.explained_variance_ratio_

    # Age groups
    bins = [15, 25, 35, 45, 55, 65, 75, 100]
    labels = ["15-25", "26-35", "36-45", "46-55", "56-65", "66-75", "75+"]
    samples["AgeGroup"] = pd.cut(samples["Age"], bins=bins, labels=labels)

    # -------------------------------
    # PCA plots
    # -------------------------------

    plot_scatter(samples, pc1_var, pc2_var, "Location")
    plot_scatter(samples, pc1_var, pc2_var, "Sex")
    plot_scatter(samples, pc1_var, pc2_var, "AgeGroup", palette="coolwarm")
    plot_scatter(samples, pc1_var, pc2_var, "lowLeuko")
    plot_scatter(samples, pc1_var, pc2_var, "mediumLeuko")
    plot_scatter(samples, pc1_var, pc2_var, "highLeuko")

    plot_boxplot(samples)

    # -------------------------------
    # Correlation analysis
    # -------------------------------

    samples["Sex_code"] = samples["Sex"].astype("category").cat.codes
    samples["Location_code"] = samples["Location"].astype("category").cat.codes

    cols = [
        "PC1", "PC2", "Age",
        "Sex_code", "Location_code",
        "lowLeuko", "mediumLeuko", "highLeuko"
    ]

    pearson = samples[cols].corr(method="pearson")
    spearman = samples[cols].corr(method="spearman")

    plot_correlation_heatmap(pearson, "Pearson")
    plot_correlation_heatmap(spearman, "Spearman")
