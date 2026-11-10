# scGSHAPA: Gene Set and Gene-Level Analysis for Single-Cell RNA-seq Using SHAP Values

scGSHAPA trains a tree-based or neural classifier (Random Forest, XGBoost, or a small neural
network) to predict phenotype (e.g. Case vs. Control) from single-cell gene expression, then
uses SHAP values from that model to score genes' contribution to the phenotype. Significance
is assessed with a phenotype-label permutation test, at two levels:

- **Gene-set / pathway level** (`explain_gene_sets`) — permutation-tests whole gene sets
  (e.g. KEGG pathways), similar in spirit to GSEA.
- **Gene level** (`explain_genes`) — a DEG/limma-style table (signed effect size, direction,
  FDR-corrected p-value) for every gene individually, without going through pathways at all.

Both return the same result schema and share built-in plotting: a volcano plot and expression
heatmap for gene-level results, and dot/bar plots for gene-set-level results.

**Gene-set analysis is entirely optional.** If you only want differential-expression-style
gene-level analysis, `fit` → `shap` → `explain_genes` is the whole pipeline — no `gene_sets`
dict required, and `explain_gene_sets` never needs to be called.

### Installation

```
pip install scGSHAPA
```

(not yet published — see Status below)

### Quickstart

```python
from scGSHAPA import scGSHAPA3
from sklearn.ensemble import RandomForestClassifier

g = scGSHAPA3(RandomForestClassifier(n_estimators=1000, n_jobs=-1), test_set_size=0.2, random_state=0)
g.fit(X, y)                                   # X: cells x genes, y: phenotype labels
g.shap(g.X_test, y_explain=g.y_test)

# Gene-level (DEG-style) analysis -- this is all you need if you don't want pathway analysis
gene_results = g.explain_genes("Case", n_tests=1000)
g.plot_genes(gene_results)                    # volcano plot
g.plot_heatmap(gene_results)                  # top-genes expression heatmap

# Gene-set / pathway-level analysis (optional, only if you also want this)
gene_set_results = g.explain_gene_sets(gene_sets, "Case", n_tests=1000)
g.plot_gene_sets_dot(gene_set_results)         # dot plot of top pathways
g.plot_gene_sets_bar(gene_set_results)         # bar plot of significant pathways
```



### Authors

- Sara Khademioureh: khademio AT ualberta.ca
- Sergio Peignier: sergio.peignier AT insa-lyon.fr
- Michael Przystupa
- Irina Dinu
- Olga Tsoy


### Maintainer

- Sara Khademioureh
