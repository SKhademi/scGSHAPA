"""
Sanity checks for GSHAPA3 on small synthetic data: gene-level explain_genes() against the
gene-set-level pval_gene_set_enrichment() it's built on, and a smoke test for the plotting
methods. Mirrors the manual checks originally run in tutorials/tests_explain_genes.ipynb in
the parent project.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gshapa import GSHAPA3


@pytest.fixture(scope="module")
def fitted_gshapa():
    """200 cells x 50 genes; genes 0-3 are made predictive of phenotype, the rest are noise."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 200, 50
    de_genes = ["gene_0", "gene_1", "gene_2", "gene_3"]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    phenotype = np.array(["Case"] * (n_cells // 2) + ["Control"] * (n_cells // 2))
    rng.shuffle(phenotype)

    X = pd.DataFrame(rng.normal(size=(n_cells, n_genes)), columns=gene_names)
    case_mask = phenotype == "Case"
    for gene in de_genes:
        X.loc[case_mask, gene] += 2.0
    y = pd.Series(phenotype)

    model = RandomForestClassifier(random_state=0, max_features=1. / 2, ccp_alpha=0.01,
                                    n_estimators=300, n_jobs=-1)
    g = GSHAPA3(model, test_set_size=0.4, random_state=0)
    g.fit(X, y)
    g.shap(g.X_test, y_explain=g.y_test)
    return g, gene_names, de_genes


def test_explain_genes_schema(fitted_gshapa):
    g, gene_names, _ = fitted_gshapa
    res = g.explain_genes("Case", n_tests=500)
    assert set(res.index) == set(gene_names)
    assert list(res.columns) == ["signed score", "direction", "p_pos corrected", "p_neg corrected"]


def test_explain_genes_score_matches_direct_computation(fitted_gshapa):
    g, gene_names, _ = fitted_gshapa
    res = g.explain_genes("Case", n_tests=500)
    direct_scores = g.compute_target_global_gene_enrichment_score("Case")
    diff = (res["signed score"] - direct_scores.reindex(res.index)).abs().max()
    assert diff < 1e-10


def test_explain_genes_separates_de_from_noise(fitted_gshapa):
    g, gene_names, de_genes = fitted_gshapa
    res = g.explain_genes("Case", n_tests=500)
    min_p = res[["p_pos corrected", "p_neg corrected"]].min(axis=1)
    noise_genes = ["gene_10", "gene_20", "gene_30"]

    assert (min_p.loc[de_genes] < 0.05).all()
    assert (min_p.loc[noise_genes] >= 0.05).all()


def test_explain_genes_agrees_with_pval_gene_set_enrichment_singletons(fitted_gshapa):
    g, gene_names, de_genes = fitted_gshapa
    res = g.explain_genes("Case", n_tests=500)
    direct_scores = g.compute_target_global_gene_enrichment_score("Case")

    for gene in de_genes:
        _, _, _, pp, pn = g.pval_gene_set_enrichment(direct_scores, [gene], n_tests=500)
        singleton_min_p = min(pp, pn)
        batch_min_p = min(res.loc[gene, "p_pos corrected"], res.loc[gene, "p_neg corrected"])
        assert (singleton_min_p < 0.05) == (batch_min_p < 0.05)


def test_plot_genes_smoke(fitted_gshapa):
    g, _, _ = fitted_gshapa
    res = g.explain_genes("Case", n_tests=200)
    ax = g.plot_genes(res, alpha=0.05, label_top_n=5)
    assert ax is not None


def test_plot_gene_sets_smoke(fitted_gshapa):
    g, gene_names, de_genes = fitted_gshapa
    remaining = [gene for gene in gene_names if gene not in de_genes]
    gene_sets = {"DE_set": de_genes + remaining[:2]}
    for i in range(3):
        gene_sets[f"noise_set_{i}"] = remaining[2 + i * 6: 2 + (i + 1) * 6]

    res = g.explain_gene_sets(gene_sets, "Case", n_tests=200)
    ax_dot = g.plot_gene_sets_dot(res, top_n=4)
    assert ax_dot is not None

    ax_bar = g.plot_gene_sets_bar(res, top_n=4, alpha=0.05)
    assert ax_bar is not None
