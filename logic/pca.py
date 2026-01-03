import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent


def plot_pca(samples, pc1, pc2, data_col, palette=None):
    """   
    samples : pd.DataFrame
        DataFrame containing 'PC1', 'PC2' and the hue_col.
    pca : sklearn PCA object
        Fitted PCA to extract explained variance for axis labels.
    hue_col : str
        Column name to color the points by.
    outdir : Path
        Directory to save the plot.
    title : str, optional
        Plot title. Defaults to "PCA by {hue_col}".
    palette : str or dict, optional
        Color palette for seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=samples,
        x="PC1",
        y="PC2",
        hue=data_col,
        palette=palette,
        s=40
    )

    # Add dotted lines at x=0 and y=0
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)

    plt.xlabel(f"PC1 ({pc1*100:.1f}%)")
    plt.ylabel(f"PC2 ({pc2*100:.1f}%)")
    plt.title(f"PCA of RNA-seq samples by {data_col}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "pca_images" / f"colored_by_{data_col}.png")
    plt.close()


if __name__ == '__main__':

    # Load data
    samples = pd.read_csv(ROOT_DIR / "data/samples.csv",
                          sep=";",
                          decimal=",",
                          index_col=0)
    numeric_cols = ["Age", "lowLeuko", "mediumLeuko", "highLeuko"]
    for col in numeric_cols:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")

    counts = pd.read_csv(ROOT_DIR / "data/counts_normalized.csv",
                         sep=";",
                         decimal=",",
                         index_col=0)

    # Match and order samples
    samples = samples[samples["id"].isin(counts.columns)]
    X = counts[samples["id"]].T

    # Center data and run PCA
    X_centered = StandardScaler(with_std=False).fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_centered)

    # Store results
    pc1 = pca.explained_variance_ratio_[0]
    pc2 = pca.explained_variance_ratio_[1]

    # Create a new column with age groups
    bins = [15, 25, 35, 45, 55, 65, 75, 100]
    labels = ["15-25", "26-35", "36-45", "46-55", "56-65", "66-75", "75+"]
    samples["AgeGroup"] = pd.cut(samples["Age"],
                                 bins=bins,
                                 labels=labels,
                                 right=True)

    plot_pca(samples, pc1, pc2, data_col="Location")
    plot_pca(samples, pc1, pc2, data_col="Sex")
    plot_pca(samples, pc1, pc2, data_col="AgeGroup", palette="coolwarm")
    plot_pca(samples, pc1, pc2, data_col="lowLeuko")
    plot_pca(samples, pc1, pc2, data_col="mediumLeuko")
    plot_pca(samples, pc1, pc2, data_col="highLeuko")

    outdir = ROOT_DIR/ "pca_images"

    # Boxplots of PC1 and PC2 by AgeGroup
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x="AgeGroup", y="PC1", data=samples)
    plt.title("PC1 by AgeGroup")
    plt.subplot(1, 2, 2)
    sns.boxplot(x="AgeGroup", y="PC2", data=samples)
    plt.title("PC2 by AgeGroup")
    plt.tight_layout()
    plt.savefig(outdir / "boxplots_by_AgeGroup.png")
    plt.close()
