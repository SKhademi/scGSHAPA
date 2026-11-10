import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


class PlottingMixin:
    """
    Plotting methods for scGSHAPA3 results.

    Color roles are fixed across all plots: "toward" the target class is blue, "away from"
    it is red, and "not significant" is neutral gray.
    """
    _PLOT_TOWARD_COLOR = "#2a78d6"   # blue
    _PLOT_AWAY_COLOR = "#e34948"     # red
    _PLOT_NONSIG_COLOR = "#c3c2b7"   # neutral gray
    _PLOT_SURFACE = "#fcfcfb"
    _PLOT_INK = "#0b0b0b"
    _PLOT_MUTED = "#898781"
    _PLOT_GRID = "#e1e0d9"

    @staticmethod
    def _combined_pvalue(res):
        """
        Two-sided combined p-value from a result table: 2*min(p_pos corrected, p_neg corrected),
        clipped to 1.
        """
        return (2 * res[["p_pos corrected", "p_neg corrected"]].min(axis=1)).clip(upper=1.0)

    def plot_genes(self, res=None, alpha=0.01, label_top_n=15, ax=None):
        """
        Volcano plot of gene-level results from explain_genes(): signed SHAP contrast score
        (x) vs. -log10(combined p-value) (y), colored by direction, with the most significant
        genes labeled by name.

        Args:
            res (pd.DataFrame): output of explain_genes(). Defaults to the result of the most
                recent explain_genes(...) call on this instance.
            alpha (float): significance threshold, drawn as a dashed reference line.
            label_top_n (int): number of most-significant genes to label with their gene id.
            ax (matplotlib.axes.Axes): axes to draw on; a new figure is created if None.

        Returns:
            matplotlib.axes.Axes
        """
        if res is None:
            res = getattr(self, "gene_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_genes(...) first, or pass res=.")

        pvals = self._combined_pvalue(res)
        # Floor zero p-values one order of magnitude below min(alpha, smallest nonzero p), so
        # they stay above the alpha line without stretching the y-axis.
        nonzero = pvals[pvals > 0]
        floor_candidates = [alpha] + ([nonzero.min()] if len(nonzero) > 0 else [])
        floor = min(floor_candidates) / 10
        y = -np.log10(pvals.clip(lower=floor))
        significant = pvals < alpha
        toward = significant & res["direction"].str.startswith("toward_")
        away = significant & res["direction"].str.startswith("away_from_")
        nonsig = ~significant

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        ax.set_facecolor(self._PLOT_SURFACE)

        ax.scatter(res.loc[nonsig, "signed score"], y[nonsig], s=16, c=self._PLOT_NONSIG_COLOR,
                   edgecolors="none", alpha=0.8, label="not significant")
        ax.scatter(res.loc[away, "signed score"], y[away], s=22, c=self._PLOT_AWAY_COLOR,
                   edgecolors="none", label="away from target class")
        ax.scatter(res.loc[toward, "signed score"], y[toward], s=22, c=self._PLOT_TOWARD_COLOR,
                   edgecolors="none", label="toward target class")

        ax.axhline(-np.log10(alpha), color=self._PLOT_MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"alpha={alpha}", xy=(1, -np.log10(alpha)), xycoords=("axes fraction", "data"),
                   color=self._PLOT_MUTED, va="bottom", ha="right", fontsize=8)

        # Stagger labels by rank so near-coincident points don't overlap.
        top_genes = pvals[significant].sort_values().index[:label_top_n]
        for i, gene in enumerate(top_genes):
            dx = 6 if i % 2 == 0 else -6
            dy = 6 + (i // 2) * 9
            ha = "left" if dx > 0 else "right"
            ax.annotate(str(gene), (res.loc[gene, "signed score"], y[gene]), fontsize=7,
                       color=self._PLOT_INK, ha=ha, xytext=(dx, dy), textcoords="offset points",
                       arrowprops=dict(arrowstyle="-", color=self._PLOT_MUTED, linewidth=0.5))

        ax.set_xlabel("Signed SHAP contrast score (Case - Control)", color=self._PLOT_INK)
        ax.set_ylabel("-log10(p-value)", color=self._PLOT_INK)
        ax.tick_params(colors=self._PLOT_MUTED)
        ax.grid(True, color=self._PLOT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=self._PLOT_INK)
        return ax

    def plot_gene_sets_dot(self, res=None, top_n=20, ax=None):
        """
        Dot plot of the top_n gene sets from explain_gene_sets(), ranked by |signed score|:
        one row per gene set, dot color by direction, dot size by significance (larger =
        smaller combined p-value).

        Args:
            res (pd.DataFrame): output of explain_gene_sets(). Defaults to the result of the
                most recent explain_gene_sets(...) call on this instance.
            top_n (int): number of top gene sets to show.
            ax (matplotlib.axes.Axes): axes to draw on; a new figure is created if None.

        Returns:
            matplotlib.axes.Axes
        """
        if res is None:
            res = getattr(self, "gene_set_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_gene_sets(...) first, or pass res=.")

        # smallest-to-largest so the strongest hit ends up at the top of the plot
        top = res.head(top_n).iloc[::-1]
        neg_log_p = -np.log10(self._combined_pvalue(top).clip(lower=np.finfo(float).tiny))
        sizes = 20 + 180 * (neg_log_p / max(1e-9, neg_log_p.max()))
        colors = np.where(top["direction"].str.startswith("toward_"),
                          self._PLOT_TOWARD_COLOR, self._PLOT_AWAY_COLOR)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
        ax.set_facecolor(self._PLOT_SURFACE)
        ax.scatter(top["signed score"], range(len(top)), s=sizes, c=colors, edgecolors="none")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8, color=self._PLOT_INK)
        ax.axvline(0, color=self._PLOT_MUTED, linewidth=1)
        ax.set_xlabel("Signed SHAP contrast score", color=self._PLOT_INK)
        ax.tick_params(colors=self._PLOT_MUTED)
        ax.grid(True, axis="x", color=self._PLOT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        legend_elems = [
            Line2D([0], [0], marker="o", linestyle="", color=self._PLOT_TOWARD_COLOR, label="toward target class"),
            Line2D([0], [0], marker="o", linestyle="", color=self._PLOT_AWAY_COLOR, label="away from target class"),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc="lower right", fontsize=8, labelcolor=self._PLOT_INK)
        return ax

    def plot_gene_sets_bar(self, res=None, top_n=20, alpha=0.01, ax=None):
        """
        Horizontal diverging bar plot of the top_n significant gene sets from
        explain_gene_sets() (combined p-value < alpha), ranked by |signed score|, colored by
        direction.

        Args:
            res (pd.DataFrame): output of explain_gene_sets(). Defaults to the result of the
                most recent explain_gene_sets(...) call on this instance.
            top_n (int): number of gene sets to show.
            alpha (float): significance threshold used to filter which gene sets are shown.
            ax (matplotlib.axes.Axes): axes to draw on; a new figure is created if None.

        Returns:
            matplotlib.axes.Axes
        """
        if res is None:
            res = getattr(self, "gene_set_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_gene_sets(...) first, or pass res=.")

        pvals = self._combined_pvalue(res)
        sig = res[pvals < alpha]
        if len(sig) == 0:
            raise ValueError(f"No gene sets significant at alpha={alpha} to plot.")
        top = sig.head(top_n).iloc[::-1]
        colors = np.where(top["direction"].str.startswith("toward_"),
                          self._PLOT_TOWARD_COLOR, self._PLOT_AWAY_COLOR)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
        ax.set_facecolor(self._PLOT_SURFACE)
        ax.barh(range(len(top)), top["signed score"], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8, color=self._PLOT_INK)
        ax.axvline(0, color=self._PLOT_MUTED, linewidth=1)
        ax.set_xlabel("Signed SHAP contrast score", color=self._PLOT_INK)
        ax.tick_params(colors=self._PLOT_MUTED)
        ax.grid(True, axis="x", color=self._PLOT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        legend_elems = [
            Line2D([0], [0], marker="s", linestyle="", color=self._PLOT_TOWARD_COLOR, label="toward target class"),
            Line2D([0], [0], marker="s", linestyle="", color=self._PLOT_AWAY_COLOR, label="away from target class"),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc="lower right", fontsize=8, labelcolor=self._PLOT_INK)
        return ax

    def plot_heatmap(self, res=None, top_n=30, X=None, y=None, ax=None):
        """
        Heatmap of the top_n genes from explain_genes() (ranked by |signed score|): rows are
        genes, columns are explained cells grouped by phenotype, values are each gene's
        expression z-scored across the explained cells (not SHAP values).

        Args:
            res (pd.DataFrame): output of explain_genes(). Defaults to the result of the
                most recent explain_genes(...) call on this instance.
            top_n (int): number of top genes (by |signed score|) to show.
            X (pd.DataFrame): cells x genes expression matrix to read values from. Defaults
                to self.X_explain (set by shap(...)).
            y (array-like): phenotype label per cell, aligned with X's rows. Defaults to
                self.y_explain (set by shap(...)).
            ax (matplotlib.axes.Axes): axes to draw the heatmap body on; a new figure (with
                a phenotype annotation strip above it) is created if None.

        Returns:
            matplotlib.axes.Axes: the heatmap body axes.
        """
        if res is None:
            res = getattr(self, "gene_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_genes(...) first, or pass res=.")
        if X is None:
            X = getattr(self, "X_explain", None)
            if X is None:
                raise ValueError("No expression data available. Call shap(...) first, or pass X=.")
        if y is None:
            y = getattr(self, "y_explain", None)
            if y is None:
                raise ValueError("No phenotype labels available. Call shap(...) first, or pass y=.")

        top_genes = [g for g in res.head(top_n).index if g in X.columns]
        y = np.asarray(y)
        order = np.argsort(y, kind="stable")  # groups cells by phenotype label
        groups = y[order]
        Z = X[top_genes].values[order].T.astype(float)  # genes x cells, phenotype-ordered

        # z-score each gene (row) across the explained cells
        mu = Z.mean(axis=1, keepdims=True)
        sigma = Z.std(axis=1, keepdims=True)
        sigma[sigma == 0] = 1.0
        Z = (Z - mu) / sigma

        # Blue-low / red-high; here color encodes expression, not direction of significance.
        cmap = LinearSegmentedColormap.from_list(
            "scgshapa_diverging", [self._PLOT_TOWARD_COLOR, "#f0efec", self._PLOT_AWAY_COLOR]
        )
        vmax = float(np.nanpercentile(np.abs(Z), 98)) or 1.0

        if ax is None:
            fig, (ax_strip, ax) = plt.subplots(
                2, 1, figsize=(8, 0.28 * len(top_genes) + 1.5),
                gridspec_kw={"height_ratios": [1, max(3, 0.28 * len(top_genes)) / 0.4]},
                sharex=True,
            )
        else:
            ax_strip = None

        im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="none")
        ax.set_yticks(range(len(top_genes)))
        ax.set_yticklabels(top_genes, fontsize=7, color=self._PLOT_INK)
        ax.set_xticks([])
        ax.set_xlabel(f"{Z.shape[1]} cells (grouped by phenotype)", color=self._PLOT_INK)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("expression (z-score within gene)", color=self._PLOT_INK, fontsize=8)
        cbar.ax.tick_params(labelsize=7, colors=self._PLOT_MUTED)

        if ax_strip is not None:
            unique_groups = list(dict.fromkeys(groups))  # stable order of first appearance
            strip_colors = ["#eb6834", self._PLOT_MUTED]
            group_to_color = {g: strip_colors[i % len(strip_colors)] for i, g in enumerate(unique_groups)}
            strip = np.array([[list(int(group_to_color[g].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                                for g in groups]]) / 255.0
            ax_strip.imshow(strip, aspect="auto", interpolation="none")
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            for spine in ax_strip.spines.values():
                spine.set_visible(False)
            legend_elems = [
                Line2D([0], [0], marker="s", linestyle="", color=group_to_color[g], label=str(g))
                for g in unique_groups
            ]
            ax_strip.legend(handles=legend_elems, frameon=False, loc="upper center",
                            bbox_to_anchor=(0.5, 2.2), ncol=len(unique_groups), fontsize=8,
                            labelcolor=self._PLOT_INK)

        return ax
