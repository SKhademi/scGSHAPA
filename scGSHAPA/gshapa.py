import math
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import fasttreeshap
from sklearn.ensemble import IsolationForest
import shap
from scGSHAPA.neural_network import NeuralNetworkTrainer
from scGSHAPA.plotting import PlottingMixin
import torch

RANDOM_STATE = 0
TEST_SET_SIZE = 0.2


def _default_model(random_state):
    return RandomForestClassifier(random_state=random_state, max_features=1./2,
                                  ccp_alpha=0.01, n_estimators=1000, n_jobs=-1)



class scGSHAPA3(PlottingMixin):
    """
    Class representing the scGSHAPA method

    Attributes:
        model (sklearn.base.BaseEstimator): Scikit-learn ML tree-based model used to learn to predict an phenotype (e.g., disease) from gene expression
        test_set_size (float): Relative size for test set
    """
    def __init__(self,
                 model=None,
                 test_set_size=TEST_SET_SIZE,
                 random_state = RANDOM_STATE
                ):
        """
        Constructor for scGSHAPA object.

        Args:
            model (sklearn.base.BaseEstimator): Scikit-learn ML model used to learn to predict an phenotype (e.g., disease) from gene expression. Defaults to a fresh RandomForestClassifier per instance.
            test_set_size (float): Relative size for test set
            random_state (int): Random seed
        """
        self.model = model if model is not None else _default_model(random_state)
        self.test_set_size = test_set_size
        self.random_state = random_state
        self.pseudo_random_generator = np.random.default_rng(seed=random_state)

    def fit(self, X, y):
        """
        Splits dataset into train and test and fits the ML model

        Args:
            X (array-like) : The training input samples matrix of shape (n_samples, n_features)
            y (pd.Series or np.array) : The target values (e.g., class labels) as integers/strings (classification) or floats (regression)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y,
                                                                                shuffle=True,
                                                                                stratify=y,
                                                                                test_size=self.test_set_size,
                                                                                random_state = self.random_state,
                                                                               )
        if isinstance(self.model, RandomForestClassifier):
            self.model.fit(self.X_train,self.y_train)
            self.le = None
        elif isinstance(self.model, XGBClassifier):
            le = LabelEncoder()
            y_train_enc = le.fit_transform(self.y_train)
            self.le = le
            self.model.fit(self.X_train, y_train_enc)
        elif isinstance(self.model, NeuralNetworkTrainer):
            le = LabelEncoder()
            y_train_enc = le.fit_transform(self.y_train)
            self.le = le

            y_train_enc = le.fit_transform(self.y_train)
            y_train_enc = torch.from_numpy(y_train_enc).long()
            X_train = torch.from_numpy(self.X_train.values).float()
            self.model = self.model.fit(X_train, y_train_enc)
            #needed for later in code
            self.model.classes_ = le.classes_

    def shap(self,X_explain=None,y_explain=None,**fasttreeshap_TreeExplainer_params):
        """
        Compute SHAP values

        Args:
            X_explain (array-like) : The explanation target samples matrix of shape (n_samples_to_explain, n_features).
                Should include both Case and Control cells (e.g. the full test set).
            y_explain (array-like) : True phenotype labels for X_explain, aligned by row order (e.g. "Case"/"Control").
            fasttreeshap_TreeExplainer_params (dict) : Parameters for fasttreeshap.TreeExplainer
        """
        self.X_explain = X_explain
        self.y_explain = y_explain
        check_addivity = False
        if isinstance(self.model, RandomForestClassifier):
            try:
                self.explainer = fasttreeshap.TreeExplainer(self.model,
                                                            **fasttreeshap_TreeExplainer_params,
                                                    )
            except Exception as e:
                print("Error in fasttreeshap.TreeExplainer, falling back to shap.TreeExplainer")
                self.explainer = shap.TreeExplainer(self.model, **fasttreeshap_TreeExplainer_params)
        elif isinstance(self.model, XGBClassifier):
            booster = self.model.get_booster()
            self.explainer = shap.TreeExplainer(booster, **fasttreeshap_TreeExplainer_params)

        elif isinstance(self.model, NeuralNetworkTrainer):
            model_for_shap = self.model.get_model_for_shap()
            # Subsample the DeepExplainer background: its cost scales with background size,
            # and 100 reference samples is enough for stable attributions.
            background_size = min(100, len(self.X_train))
            background_idx = np.random.RandomState(self.random_state).choice(
                len(self.X_train), size=background_size, replace=False
            )
            X_train = torch.from_numpy(self.X_train.values[background_idx]).float()
            self.explainer = shap.DeepExplainer(model_for_shap, X_train, **fasttreeshap_TreeExplainer_params)
            X_explain = torch.from_numpy(self.X_explain.values).float()
            check_addivity = False

        else:
            raise ValueError("Model type not supported. Please use RandomForestClassifier, XGBClassifier or NeuralNetworkTrainer.")

        if isinstance(self.model, NeuralNetworkTrainer):
            # Raw DeepExplainer output (list of arrays or 3D array); class extraction happens
            # in compute_target_global_gene_enrichment_score.
            self.SHAP_values = self.explainer.shap_values(X_explain, check_additivity=check_addivity)
        else:
            self.SHAP_values = self.explainer(X_explain, check_additivity=check_addivity)


    def compute_target_global_gene_enrichment_score(self, target_class):
        """
        Compute the gene-level enrichment scores for X_explain and the class of interest.

        Args:
            target_class (int or str) : label for the class for which the gene global enrichment scores should be computed

        Returns:
            pd.Series: gene-level scores, mean(SHAP[Case cells]) - mean(SHAP[Control cells]).
                Requires self.y_explain (set by shap()) to cover both groups.
        """
        if isinstance(self.model, NeuralNetworkTrainer):
            # The network has a single output column: the logit of P(encoded class 1).
            # LabelEncoder sorts labels, so Case=0, Control=1; negate when the target is class 0.
            target_encoded = list(self.le.classes_).index(target_class)
            if isinstance(self.SHAP_values, list):
                if len(self.SHAP_values) == 1:
                    shap_arr = np.array(self.SHAP_values[0])
                    if target_encoded == 0:
                        shap_arr = -shap_arr
                else:
                    # Genuine per-class output: take the target class index directly.
                    shap_arr = np.array(self.SHAP_values[target_encoded])
            elif isinstance(self.SHAP_values, np.ndarray) and self.SHAP_values.ndim == 3:
                if self.SHAP_values.shape[-1] == 1:
                    shap_arr = self.SHAP_values[:, :, 0]
                    if target_encoded == 0:
                        shap_arr = -shap_arr
                else:
                    # Genuine per-class output: take the target class index directly.
                    shap_arr = self.SHAP_values[:, :, target_encoded]
            else:
                # Single 2D array: same single-output convention as above.
                shap_arr = np.array(self.SHAP_values)
                if target_encoded == 0:
                    shap_arr = -shap_arr
            shap_values_interest = pd.DataFrame(shap_arr,
                                                index=self.X_explain.index,
                                                columns=self.X_explain.columns)
        elif isinstance(self.model, XGBClassifier):
            # The raw XGBoost booster predicts log-odds for encoded class 1 (binary only).
            assert len(self.le.classes_) == 2, (
                f"XGBoost direction flip only supports binary classification, "
                f"but got classes: {list(self.le.classes_)}"
            )
            shap_arr = self.SHAP_values.values if hasattr(self.SHAP_values, 'values') else np.array(self.SHAP_values)
            shap_values_interest = pd.DataFrame(shap_arr,
                                                index=self.X_explain.index,
                                                columns=self.X_explain.columns)
            target_encoded = list(self.le.classes_).index(target_class)
            if target_encoded == 0:
                shap_values_interest = -shap_values_interest
        elif len(self.SHAP_values.shape) == 3:
            # RF (fasttreeshap): 3D (n_samples, n_features, n_classes), model.classes_ has string labels
            index_target_class = list(self.model.classes_).index(target_class)
            shap_values_interest = pd.DataFrame(self.SHAP_values.values[:,:,index_target_class],
                                                index=self.X_explain.index,
                                                columns=self.X_explain.columns)
        else:
            shap_values_interest = pd.DataFrame(self.SHAP_values.values,
                                                index=self.X_explain.index,
                                                columns=self.X_explain.columns)
        if self.y_explain is None:
            raise ValueError(
                "y_explain was not set. Call shap(X_explain, y_explain=<labels for X_explain>) "
                "with labels covering both Case and Control before computing the gene score."
            )
        case_mask = np.asarray(self.y_explain) == "Case"
        control_mask = np.asarray(self.y_explain) == "Control"
        if not case_mask.any():
            raise ValueError("y_explain contains no 'Case' cells - gene_score needs both groups.")
        if not control_mask.any():
            raise ValueError("y_explain contains no 'Control' cells - gene_score needs both groups.")
        # Cached for the phenotype-label permutation null in pval_gene_set_enrichment/explain_genes.
        self._shap_values_interest = shap_values_interest
        self._case_mask = case_mask
        self._control_mask = control_mask
        global_gene_enrichment_score = (
            shap_values_interest[case_mask].mean() - shap_values_interest[control_mask].mean()
        )
        global_gene_enrichment_score_raw = global_gene_enrichment_score.copy()
        self.global_feature_importance = global_gene_enrichment_score_raw.sort_values(ascending=False)
        return global_gene_enrichment_score

    def pval_gene_set_enrichment(self, global_gene_enrichment_score, set_of_genes, n_tests=10000):
        """
        Sample the signed permutation null once and derive all p-values from it.

        Args:
            global_gene_enrichment_score (pd.Series): signed gene-level SHAP scores from
                compute_target_global_gene_enrichment_score
            set_of_genes (list): gene ids in the gene set
            n_tests (int): number of permutations

        Returns:
            (unsigned_score, p_u, signed_score, p_pos, p_neg)
                unsigned_score = abs(mean(shap[genes]))
                p_u            = P(|null| >= unsigned_score)   [magnitude test]
                signed_score   = mean(shap[genes])
                p_pos          = P(null >= signed_score)        [enriched toward target class]
                p_neg          = P(null <= signed_score)        [enriched away from target class]

        Note:
            The null is built by permuting the Case/Control labels of the explained cells
            (a self-contained test; Goeman & Buhlmann, 2007), holding group sizes fixed.
        """
        def sampling(genes, n_tests):
            cell_scores = self._shap_values_interest[genes].mean(axis=1).values
            n_cells = cell_scores.shape[0]
            n_case = int(self._case_mask.sum())
            res = np.empty(n_tests)
            for i in range(n_tests):
                perm = self.pseudo_random_generator.permutation(n_cells)
                res[i] = cell_scores[perm[:n_case]].mean() - cell_scores[perm[n_case:]].mean()
            return res

        nb_genes = len(set_of_genes)

        if nb_genes == 0:
            return 0.0, 1.0, 0.0, 1.0, 1.0

        signed_score = global_gene_enrichment_score[set_of_genes].mean()
        unsigned_score = np.abs(signed_score)

        # Sample the signed null once; derive both tests from the same draws.
        sample = sampling(set_of_genes, n_tests)

        # Permutation p-values with the +1 correction (Phipson & Smyth, 2010): p >= 1/(n_tests+1), never 0.
        # Unsigned test: null = abs(mean(shap_perm)), matches observed statistic abs(mean(shap[genes]))
        p_u = (1 + np.sum(np.abs(sample) >= unsigned_score)) / (n_tests + 1)

        # Signed tests: upper/lower tails of the signed null
        p_pos = (1 + np.sum(sample >= signed_score)) / (n_tests + 1)  # enriched toward target class
        p_neg = (1 + np.sum(sample <= signed_score)) / (n_tests + 1)  # enriched away from target class

        return unsigned_score, p_u, signed_score, p_pos, p_neg


    def explain_gene_sets(self,
                          gene_sets,
                          target_class,
                          n_tests=1000,
                          alpha=0.01,
                          multiple_test_method="fdr_bh",
                          ):
        """
        Computes signed gene set enrichment scores, estimates their statistical significance,
        and applies multiple test correction separately to each p-value family (p_pos, p_neg).

        Args:
            gene_sets (dict): keys are gene set names, values are lists of gene ids
            target_class (int or str): label for the class of interest
            n_tests (int): number of permutations per gene set
            alpha (float): significance threshold for multiple test correction
            multiple_test_method (str): correction method from statsmodels.stats.multitest.multipletests

        Returns:
            pd.DataFrame: one row per gene set, sorted by |signed score|, columns:
                - signed score   : mean(gene contrast scores) = mean(SHAP[Case])-mean(SHAP[Control])
                - direction      : "toward_<class>" if signed > 0 else "away_from_<class>"
                - p_pos corrected: FDR-corrected P(null >= signed_score) [enriched toward target class]
                - p_neg corrected: FDR-corrected P(null <= signed_score) [enriched away from target class]
              Call a gene set significant when min(p_pos corrected, p_neg corrected) < alpha;
              `direction` says which one to look at.
        """
        global_gene_enrichment_score = self.compute_target_global_gene_enrichment_score(target_class)

        # Seed once per call: reproducible under self.random_state, distinct draws per gene set.
        self.pseudo_random_generator = np.random.default_rng(seed=self.random_state)

        # Permutation resolution is bounded by the number of distinct relabelings of the cells.
        n_cells_explained = int(self._case_mask.shape[0])
        n_case_explained = int(self._case_mask.sum())
        distinct_relabelings = math.comb(n_cells_explained, n_case_explained)
        if distinct_relabelings < 10 * n_tests:
            print(
                f"WARNING: only {distinct_relabelings:,} distinct Case/Control relabelings of "
                f"{n_cells_explained} explained cells are possible, vs n_tests={n_tests} requested - "
                f"permutation p-value resolution may be coarse at low alpha. Consider a larger "
                f"explained cell count (e.g. test_set_size) or a smaller n_tests."
            )

        signed_scores, p_pos, p_neg = {}, {}, {}

        for g_s in tqdm(gene_sets):
            genes = list(set(gene_sets[g_s]).intersection(self.X_explain.columns))

            _, _, s_score, pp, pn = self.pval_gene_set_enrichment(
                global_gene_enrichment_score, genes, n_tests)

            signed_scores[g_s] = s_score
            p_pos[g_s] = pp
            p_neg[g_s] = pn

        p_pos = pd.Series(p_pos)
        p_neg = pd.Series(p_neg)

        _, p_pos_corr, _, _ = multipletests(p_pos.values, alpha=alpha, method=multiple_test_method)
        _, p_neg_corr, _, _ = multipletests(p_neg.values, alpha=alpha, method=multiple_test_method)

        signed_scores = pd.Series(signed_scores)
        direction = signed_scores.apply(
            lambda x: "toward_" + str(target_class) if x > 0 else "away_from_" + str(target_class)
        )

        res = pd.DataFrame({
            "signed score":      signed_scores,
            "direction":         direction,
            "p_pos corrected":   pd.Series(p_pos_corr, index=p_pos.index),
            "p_neg corrected":   pd.Series(p_neg_corr, index=p_neg.index),
        })
        res = res.reindex(signed_scores.abs().sort_values(ascending=False).index)
        # cached so plot_gene_sets_dot/plot_gene_sets_bar can default to "the last result"
        self.gene_set_results_ = res
        return res

    def explain_genes(self,
                      target_class,
                      n_tests=1000,
                      alpha=0.01,
                      multiple_test_method="fdr_bh",
                      ):
        """
        Per-gene analog of explain_gene_sets: computes a signed SHAP contrast score for every
        gene, estimates a phenotype-permutation p-value for each gene, and applies multiple
        test correction. Gives a DEG-style table (effect size + p-value + FDR) without going
        through gene sets at all.

        Uses the same phenotype-label permutation null as explain_gene_sets, with one shared
        set of n_tests permutations applied to all genes at once.

        Args:
            target_class (int or str): label for the class of interest
            n_tests (int): number of permutations
            alpha (float): significance threshold for multiple test correction
            multiple_test_method (str): correction method from statsmodels.stats.multitest.multipletests

        Returns:
            pd.DataFrame: one row per gene, same schema as explain_gene_sets:
                - signed score   : mean(SHAP[Case]) - mean(SHAP[Control]) for that gene
                - direction      : "toward_<class>" if signed > 0 else "away_from_<class>"
                - p_pos corrected: FDR-corrected P(null >= signed_score) [enriched toward target class]
                - p_neg corrected: FDR-corrected P(null <= signed_score) [enriched away from target class]

        Note:
            The null array is (n_tests, n_genes), so memory grows with both.
        """
        global_gene_enrichment_score = self.compute_target_global_gene_enrichment_score(target_class)

        self.pseudo_random_generator = np.random.default_rng(seed=self.random_state)

        shap_matrix = self._shap_values_interest.values  # (n_cells, n_genes)
        n_cells = shap_matrix.shape[0]
        n_case = int(self._case_mask.sum())

        distinct_relabelings = math.comb(n_cells, n_case)
        if distinct_relabelings < 10 * n_tests:
            print(
                f"WARNING: only {distinct_relabelings:,} distinct Case/Control relabelings of "
                f"{n_cells} explained cells are possible, vs n_tests={n_tests} requested - "
                f"permutation p-value resolution may be coarse at low alpha. Consider a larger "
                f"explained cell count (e.g. test_set_size) or a smaller n_tests."
            )

        # One shared set of permutations; each computes the contrast for all genes at once.
        null = np.empty((n_tests, shap_matrix.shape[1]))
        for i in range(n_tests):
            perm = self.pseudo_random_generator.permutation(n_cells)
            null[i] = shap_matrix[perm[:n_case]].mean(axis=0) - shap_matrix[perm[n_case:]].mean(axis=0)

        signed_scores = global_gene_enrichment_score
        # +1 correction, same as pval_gene_set_enrichment: p >= 1/(n_tests+1), never 0
        p_pos = pd.Series((1 + (null >= signed_scores.values[None, :]).sum(axis=0)) / (n_tests + 1),
                          index=signed_scores.index)
        p_neg = pd.Series((1 + (null <= signed_scores.values[None, :]).sum(axis=0)) / (n_tests + 1),
                          index=signed_scores.index)

        _, p_pos_corr, _, _ = multipletests(p_pos.values, alpha=alpha, method=multiple_test_method)
        _, p_neg_corr, _, _ = multipletests(p_neg.values, alpha=alpha, method=multiple_test_method)

        direction = signed_scores.apply(
            lambda x: "toward_" + str(target_class) if x > 0 else "away_from_" + str(target_class)
        )

        res = pd.DataFrame({
            "signed score":      signed_scores,
            "direction":         direction,
            "p_pos corrected":   pd.Series(p_pos_corr, index=signed_scores.index),
            "p_neg corrected":   pd.Series(p_neg_corr, index=signed_scores.index),
        })
        res = res.reindex(signed_scores.abs().sort_values(ascending=False).index)
        # cached so plot_genes can default to "the last result"
        self.gene_results_ = res
        return res

    def isolation_forest_gene_selection(self,**isolation_forest_parameters):
        '''
        Apply the Isolation Forest outlier detection algorithm on the average genes' Shap values to detect outstanding genes.
        Params:
            **isolation_forest_parameters: Parameters for the sklearn Isolation Forest class

        Returns:
            pd.DataFrame: Each row is a gene, the first column "Avg. SHAP value" depicts the average SHAP value for the given gene, and "Is Selected" is True if the gene is selected as outstanding and False otherwise

        '''
        self.gene_importance = pd.DataFrame(self.global_feature_importance,columns=["Avg. SHAP value"])
        isolation_forest_parameters["random_state"] = self.random_state
        isofo = IsolationForest(**isolation_forest_parameters)
        outliers = isofo.fit_predict(np.abs(self.gene_importance.values))
        selected = (outliers == -1)
        self.gene_importance["Is Selected"] = (outliers == -1)
        return self.gene_importance
