import pandas as pd
from collections import defaultdict
from pathlib import Path

from goatools.obo_parser import GODag
from goatools.associations import read_associations
from goatools.go_enrichment import GOEnrichmentStudy

# -----------------------------
# File paths (edit as needed)
# -----------------------------
ROOT_DIR = Path("C:/PG/sem_9/EB/RnaBiasDetection")  # adjust to your project root
CLUSTER_FILE = ROOT_DIR / "data" / "clusters.csv"
BACKGROUND_FILE = ROOT_DIR / "data" / "deg_malignant_vs_nonmalignant.csv"
OBO_FILE = ROOT_DIR / "data" / "go-basic.obo"
ENSEMBL_GO_FILE = ROOT_DIR / "data" / "ensembl_go.txt"  # BioMart mapping

# -----------------------------
# Load cluster assignments
# -----------------------------
clusters_df = pd.read_csv(CLUSTER_FILE, index_col=0)

# gene -> cluster
gene_to_cluster = clusters_df["Cluster"]

# cluster -> genes
cluster_to_genes = defaultdict(set)
for gene, cluster in gene_to_cluster.items():
    cluster_to_genes[cluster].add(gene)

# -----------------------------
# Load background genes
# -----------------------------
background_df = pd.read_csv(BACKGROUND_FILE, index_col=0)
background_genes = set(background_df.index)

print(f"Total background genes: {len(background_genes)}")

# -----------------------------
# Load GO ontology
# -----------------------------
go_dag = GODag(OBO_FILE)

# -----------------------------
# Load Ensembl → GO associations
# -----------------------------
geneid2gos = read_associations(ENSEMBL_GO_FILE, sep="\t")

# Filter to only background genes
geneid2gos = {gene: gos for gene, gos in geneid2gos.items() if gene in background_genes}

print(f"Background genes annotated with GO terms: {len(geneid2gos)}")

# -----------------------------
# Initialize GO enrichment object
# -----------------------------
goea = GOEnrichmentStudy(
    background_genes,
    geneid2gos,
    go_dag,
    methods=["fdr_bh"],  # Benjamini–Hochberg
    alpha=0.05
)

# -----------------------------
# Run enrichment per cluster
# -----------------------------
results_all_clusters = []

for cluster_id, genes in cluster_to_genes.items():
    study_genes = genes & background_genes
    if len(study_genes) == 0:
        continue

    goea_results = goea.run_study(study_genes)

    for r in goea_results:
        if r.p_fdr_bh < 0.05:
            results_all_clusters.append({
                "cluster": cluster_id,
                "GO_ID": r.GO,
                "name": r.name,
                "namespace": r.NS,  # BP, MF, CC
                "p_uncorrected": r.p_uncorrected,
                "p_fdr": r.p_fdr_bh,
                "study_count": r.study_count,
                "study_size": r.study_n,
                "population_count": r.pop_count,
                "population_size": r.pop_n
            })

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results_all_clusters)
results_df.sort_values(["cluster", "namespace", "p_fdr"], inplace=True)

results_df.to_csv(ROOT_DIR / "go_enrichment_by_cluster.csv", index=False)

print("GO enrichment completed.")
print("Results saved to: go_enrichment_by_cluster.csv")
