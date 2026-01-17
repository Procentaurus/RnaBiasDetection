# ============================================================
# RNA-seq DEGs analysis using edgeR (fixed & validated)
# Semicolon-separated CSV, comma decimal
# ============================================================

library(edgeR)

# -----------------------------
# 1. Paths
# -----------------------------
ROOT_DIR <- "C:/PG/sem_9/EB/RnaBiasDetection"
counts_file   <- file.path(ROOT_DIR, "data", "counts_raw.csv")
metadata_file <- file.path(ROOT_DIR, "data", "samples.csv")

# -----------------------------
# 2. Load data
# -----------------------------
counts <- read.csv(
  counts_file,
  sep=";",
  dec=",",
  row.names=1,
  check.names=FALSE
)

metadata <- read.csv(
  metadata_file,
  sep=";",
  dec=",",
  row.names=1,
  check.names=FALSE
)

# Ensure identical sample order
counts <- counts[, rownames(metadata)]

group <- factor(metadata$Group)

# -----------------------------
# 3. Group-aware filtering (FIX)
# -----------------------------
keep <- filterByExpr(counts, group=group)
counts_filtered <- counts[keep, ]

cat("Genes before filtering:", nrow(counts), "\n")
cat("Genes after filtering :", nrow(counts_filtered), "\n")

# -----------------------------
# 4. Create DGEList + normalize
# -----------------------------
dge <- DGEList(counts=counts_filtered, group=group)
dge <- calcNormFactors(dge)

# -----------------------------
# 5. Design matrix
# -----------------------------
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)

# -----------------------------
# 6. Dispersion estimation
# -----------------------------
dge <- estimateDisp(dge, design)

# -----------------------------
# 7. GLM fit
# -----------------------------
fit <- glmFit(dge, design)

# -----------------------------
# 8. Pairwise DE comparisons
# -----------------------------
groups <- levels(group)
results <- list()

for (i in 1:(length(groups)-1)) {
  for (j in (i+1):length(groups)) {

    contrast <- rep(0, length(groups))
    contrast[i] <- -1
    contrast[j] <- 1

    lrt <- glmLRT(fit, contrast=contrast)
    table <- topTags(
      lrt,
      n=Inf,
      adjust.method="BH",
      sort.by="PValue"
    )$table

    key <- paste(groups[j], "vs", groups[i], sep="_")
    results[[key]] <- table

    cat("\nPairwise comparison:", key, "\n")
    cat("DEGs (FDR < 0.05):", sum(table$FDR < 0.05), "\n")

  # --- 8.5. Save MA plot ---
    png(
      filename = file.path(ROOT_DIR, "images/deg", paste0("MA_", key, ".png")),
      width = 1000,
      height = 800,
      res = 120
    )

    plotMD(lrt, main = key)
    abline(h = c(-1, 1), col = "blue", lty = 2)

    dev.off()
  }
}

# -----------------------------
# 9. Save DEG tables
# -----------------------------
for (key in names(results)) {
  out_file <- file.path(ROOT_DIR, paste0("DEGs_", key, ".csv"))
  write.csv(results[[key]], out_file, row.names=TRUE)
}

# -----------------------------
# 10. Randomization control (FIX)
# -----------------------------
set.seed(123)

metadata_rand <- metadata
metadata_rand$Group <- sample(metadata_rand$Group)

group_rand <- factor(metadata_rand$Group)

dge_rand <- DGEList(counts=counts_filtered, group=group_rand)
dge_rand <- calcNormFactors(dge_rand)

design_rand <- model.matrix(~0 + group_rand)
dge_rand <- estimateDisp(dge_rand, design_rand)
fit_rand <- glmFit(dge_rand, design_rand)

contrast_rand <- c(-1, 1)
lrt_rand <- glmLRT(fit_rand, contrast=contrast_rand)

table_rand <- topTags(lrt_rand, n=Inf)$table

cat("\nRandomization test:\n")
cat("DEGs (FDR < 0.05):", sum(table_rand$FDR < 0.05), "\n")

# ============================================================
# End of script
# ============================================================
