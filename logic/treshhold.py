import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logic.params import ROOT_DIR


if __name__ == '__main__':
    counts = pd.read_csv(ROOT_DIR / "data/counts_normalized.csv",
                            sep=";",
                            decimal=",",
                            index_col=0)

    cutoffs = np.logspace(0.4, 1, 10)

    # Count genes above cutoff per sample
    gene_counts = {
        cutoff: (counts > cutoff).sum(axis=0)
        for cutoff in cutoffs
    }

    gene_counts_df = pd.DataFrame(gene_counts)

    # Plot
    plt.figure(figsize=(8, 6))
    for sample in gene_counts_df.index:
        plt.plot(
            cutoffs,
            gene_counts_df.loc[sample],
            color="gray",
            alpha=0.3
        )

    plt.xscale("log")
    plt.xlabel("Expression cutoff")
    plt.ylabel("Number of genes expressed")
    plt.title("Genes above expression cutoff per sample")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "images" / "treshhold" / "expression_cutoff_diagnostic.png")
    plt.close()
