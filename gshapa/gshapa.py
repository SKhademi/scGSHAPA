import math
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import fasttreeshap
from sklearn.ensemble import IsolationForest
#work arounds for xgboost
import shap
from gshapa.neural_network import NeuralNetworkTrainer
import torch

RANDOM_STATE = 0
TEST_SET_SIZE = 0.2


def _default_model(random_state):
    return RandomForestClassifier(random_state=random_state, max_features=1./2,
                                  ccp_alpha=0.01, n_estimators=1000, n_jobs=-1)



class GSHAPA3:
    """
    Class representing the GSHAPA method

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
        Constructor for GSHAPA object.

        Args:
            model (sklearn.base.BaseEstimator): Scikit-learn ML model used to learn to predict an phenotype (e.g., disease) from gene expression. Defaults to a fresh RandomForestClassifier per instance (never a shared default object).
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

            y_train_enc = torch.from_numpy(y_train_enc).long()
            X_train = torch.from_numpy(self.X_train.values).float()
            self.model = self.model.fit(X_train, y_train_enc)
            # sklearn-style .classes_ attribute, needed by callers (e.g. plot code) that
            # expect it regardless of which backend is in use
            self.model.classes_ = le.classes_

    def shap(self,X_explain=None,y_explain=None,**fasttreeshap_TreeExplainer_params):
        """
        Compute SHAP values

        Args:
            X_explain (array-like) : The explanation target samples matrix of shape (n_samples_to_explain, n_features).
            y_explain (array-like) : True phenotype labels for X_explain, aligned by row order (e.g. "Case"/"Control").
                Needed so compute_target_global_gene_enrichment_score can split X_explain into
                Case/Control groups for the mean(SHAP[Case]) - mean(SHAP[Control]) contrast. X_explain
                should therefore include both groups (e.g. the full test set), not a Case-only subset.
            fasttreeshap_TreeExplainer_params (dict) : Parameters for fasttreeshap.TreeExplainer
        """
        self.X_explain = X_explain
        self.y_explain = y_explain
        check_additivity = False
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
            # Subsample the DeepExplainer background instead of passing the full training
            # set. DeepExplainer's cost scales with background_size x explained_size (it does a
            # forward/backward pass per background reference per explained sample), unlike
            # TreeSHAP above, which is exact and doesn't consume a background sample at all. On
            # a ~4,900-row training split, passing the whole training set as background made a
            # single real-data fit take ~19h for just 25 gene sets - the fixed DeepExplainer
            # setup cost, not the per-gene-set permutation testing, which is fast. 100 background
            # samples is within SHAP's own documented range for DeepExplainer and leaves the
            # background a representative reference distribution, not data being explained, so
            # the underlying attributions are essentially unaffected.
            background_size = min(100, len(self.X_train))
            background_idx = np.random.RandomState(self.random_state).choice(
                len(self.X_train), size=background_size, replace=False
            )
            X_train = torch.from_numpy(self.X_train.values[background_idx]).float()
            self.explainer = shap.DeepExplainer(model_for_shap, X_train, **fasttreeshap_TreeExplainer_params)
            X_explain = torch.from_numpy(self.X_explain.values).float()

        else:
            raise ValueError("Model type not supported. Please use RandomForestClassifier, XGBClassifier or NeuralNetworkTrainer.")

        if isinstance(self.model, NeuralNetworkTrainer):
            # Store raw output from DeepExplainer (may be a list of arrays, one per class,
            # or a 3D ndarray). Class extraction is handled in compute_target_global_gene_enrichment_score.
            self.SHAP_values = self.explainer.shap_values(X_explain, check_additivity=check_additivity)
        else:
            self.SHAP_values = self.explainer(X_explain, check_additivity=check_additivity)


    def compute_target_global_gene_enrichment_score(self, target_class):
        """
        Compute the average (global) gene-level enrichment scores for X_explain objects and the class of interest (in the case of regression the target_class param is ignored)

        Args:
            target_class (int or str) : label for the class for which the gene global enrichment scores should be computed

        Returns:
            pd.Series: gene-level global enrichment scores.
                gene_score = mean(SHAP[Case cells]) - mean(SHAP[Control cells]), a two-sample
                contrast. Requires self.y_explain (set by shap()) to cover both groups.
        """
        # Model-specific SHAP extraction to correctly handle direction for each backend.
        if isinstance(self.model, NeuralNetworkTrainer):
            # DeepExplainer returns either a list [shap_class0, shap_class1, ...] (one 2D array per
            # output neuron) or a 3D ndarray (n_samples, n_features, n_classes).
            # le.classes_ is sorted alphabetically by LabelEncoder: Case=0, Control=1.
            target_encoded = list(self.le.classes_).index(target_class)
            # NeuralNetworkTrainer.fit() hard-raises above 2 classes, and
            # SimpleClassifier.forward() always collapses num_classes=2 to a single output
            # column (see neural_network.py) - GSHAPA3.fit() LabelEncodes y first and trains
            # that column with BCEWithLogitsLoss against the encoded label, so it is always the
            # logit of P(encoded class 1) = P(Control), never a genuine per-class output. With
            # shap==0.49.1, DeepExplainer wraps that single output as a 3D array shaped
            # (n_samples, n_features, 1) rather than a plain 2D array. Treat "only one output
            # column" (list of length 1, or 3D with a trailing singleton dim, or a plain 2D
            # array) as the P(Control) case uniformly and negate when target_encoded == 0; only
            # index directly by target_encoded when there are genuinely 2+ separate output
            # columns.
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
                # Single 2D array: same single-output P(Control) convention as above.
                shap_arr = np.array(self.SHAP_values)
                if target_encoded == 0:
                    shap_arr = -shap_arr
            shap_values_interest = pd.DataFrame(shap_arr,
                                                index=self.X_explain.index,
                                                columns=self.X_explain.columns)
        elif isinstance(self.model, XGBClassifier):
            # The raw XGBoost booster predicts log-odds for encoded class 1. sklearn's
            # LabelEncoder.classes_ is always np.unique(y), i.e. always already sorted
            # regardless of input order. The precondition for the "negate when
            # target_encoded == 0" trick is that this is a binary problem (a raw XGBoost
            # booster only has one output column, log-odds of encoded class 1).
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
        # gene_score = mean(SHAP[Case]) - mean(SHAP[Control]) instead of mean(SHAP) over all
        # of X_explain. Requires y_explain to contain both groups. Explicit guard: if
        # y_explain wasn't passed to shap(...), self.y_explain is None and
        # `np.asarray(None) == "Case"` silently collapses to a bare scalar False instead of a
        # boolean array. Indexing shap_values_interest with that scalar then raises a cryptic
        # `KeyError: False` deep in pandas with no indication of the real cause. Fail loudly
        # here instead.
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
        # Cache the per-cell SHAP matrix and group masks so pval_gene_set_enrichment can build
        # a phenotype-label-permutation null without recomputing or re-explaining anything - it
        # just re-aggregates these same cached values under reshuffled Case/Control labels.
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
            global_gene_enrichment_score (pd.Series): signed gene-level SHAP scores, computed
                with the TRUE Case/Control labels by compute_target_global_gene_enrichment_score
                (also the source of the observed signed_score/unsigned_score below).
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
            The null is built via phenotype-label permutation (a self-contained test: "is this
            gene set's Case-vs-Control contrast distinguishable from a random relabeling of
            these same cells?"), not gene resampling from the rest of the gene panel (a
            competitive test: "is this gene set more enriched than a random gene set from the
            genome?"). See Goeman & Buhlmann (2007) for this self-contained vs. competitive
            distinction in the gene set testing literature. A competitive (gene-resampling)
            null's reference population - the whole gene panel - inevitably includes the real
            DE genes, whose net contribution isn't exactly zero (asymmetric up/down-regulation
            effect sizes), which shifts the null's mean away from what a genuinely-null gene
            set's mean actually looks like, inflating Type I error. Phenotype-label permutation
            never looks at any gene outside the tested set, so it isn't sensitive to that.
            The permutation draws n_cells (holding group sizes fixed) instead of the gene panel,
            so resolution is governed by the number of distinct relabelings of the explained
            cells, C(n_cells, n_case) - not by n_genes. explain_gene_sets checks this is
            comfortably larger than n_tests and warns if not (see there).
            self.pseudo_random_generator is reseeded once per explain_gene_sets(...) call, not
            per gene set, so a run stays reproducible under self.random_state while each gene
            set's relabelings differ from the next.
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

        # Unsigned test: null = abs(mean(shap_perm)), matches observed statistic abs(mean(shap[genes]))
        ecdf_u = ECDF(np.abs(sample))
        p_u = 1 - ecdf_u(unsigned_score)

        # Signed tests: upper/lower tails of the signed null
        ecdf_s = ECDF(sample)
        p = ecdf_s(signed_score)
        p_pos = 1 - p   # P(null >= signed_score): enriched toward target class
        p_neg = p        # P(null <= signed_score): enriched away from target class

        return unsigned_score, p_u, signed_score, p_pos, p_neg


    def explain_gene_sets(self,
                          gene_sets,
                          target_class,
                          n_tests=1000,
                          alpha=0.01,
                          multiple_test_method="fdr_bh",
                          ):
        """
        Computes unsigned and signed gene set enrichment scores, estimates statistical
        significance for both, and applies multiple test correction separately to each
        p-value family (p_u, p_pos, p_neg).

        Args:
            gene_sets (dict): keys are gene set names, values are lists of gene ids
            target_class (int or str): label for the class of interest
            n_tests (int): number of permutations per gene set
            alpha (float): significance threshold for multiple test correction
            multiple_test_method (str): correction method from statsmodels.stats.multitest.multipletests

        Returns:
            pd.DataFrame: one row per gene set, columns:
                - signed score   : mean(gene contrast scores) = mean(SHAP[Case])-mean(SHAP[Control])
                - direction      : "toward_<class>" if signed > 0 else "away_from_<class>"
                - p_pos corrected: FDR-corrected P(null >= signed_score) [enriched toward target class]
                - p_neg corrected: FDR-corrected P(null <= signed_score) [enriched away from target class]
              Call a gene set significant when min(p_pos corrected, p_neg corrected) < alpha;
              `direction` says which one to look at.
        """
        global_gene_enrichment_score = self.compute_target_global_gene_enrichment_score(target_class)

        # Seed the permutation RNG once per explain_gene_sets(...) call instead of once per
        # gene set. Keeps the whole call reproducible under self.random_state while giving each
        # gene set its own draw.
        self.pseudo_random_generator = np.random.default_rng(seed=self.random_state)

        # pval_gene_set_enrichment's null is phenotype-label permutation over the explained
        # cells, so resolution is bounded by the number of distinct ways to relabel n_cells
        # into (n_case, n_control) groups - C(n_cells, n_case) - not by n_genes. Warn loudly if
        # that's not comfortably above n_tests, since a small explained-cell count (e.g. a
        # small test_set_size or few total cells) would coarsen p-value resolution at the
        # low-alpha end silently otherwise.
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

            # unsigned_score/p_u are still returned by pval_gene_set_enrichment but not used
            # here - discarded via `_`. `unsigned score` is just abs(signed score) - no new
            # information given the decision rule "significant if either directional corrected
            # p-value clears alpha."
            _, _, s_score, pp, pn = self.pval_gene_set_enrichment(
                global_gene_enrichment_score, genes, n_tests)

            signed_scores[g_s] = s_score
            p_pos[g_s] = pp
            p_neg[g_s] = pn

        p_pos = pd.Series(p_pos)
        p_neg = pd.Series(p_neg)

        # Each p-value family gets its own independent multiple-testing correction.
        _, p_pos_corr, _, _ = multipletests(p_pos.values, alpha=alpha, method=multiple_test_method)
        _, p_neg_corr, _, _ = multipletests(p_neg.values, alpha=alpha, method=multiple_test_method)

        signed_scores = pd.Series(signed_scores)
        # direction column so results can be filtered by significance AND direction without
        # extra post-processing.
        direction = signed_scores.apply(
            lambda x: "toward_" + str(target_class) if x > 0 else "away_from_" + str(target_class)
        )

        res = pd.DataFrame({
            "signed score":      signed_scores,
            "direction":         direction,
            "p_pos corrected":   pd.Series(p_pos_corr, index=p_pos.index),
            "p_neg corrected":   pd.Series(p_neg_corr, index=p_neg.index),
        })
        # sort by |signed score| so the largest-magnitude effects surface first
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
        gene individually, estimates a phenotype-permutation p-value for each gene, and applies
        multiple test correction. Gives a DEG/limma-style table (effect size + p-value + FDR)
        for gene-level analysis, without going through gene sets/pathways at all.

        Uses the same self-contained phenotype-label-permutation null as
        pval_gene_set_enrichment/explain_gene_sets, but draws the n_tests permutations once and
        reuses them across every gene via a single vectorized pass over the full SHAP matrix,
        instead of calling pval_gene_set_enrichment with a singleton gene list once per gene
        (which would redraw n_tests fresh permutations for every single gene - n_genes times
        more permutation draws for the same result, since the permutation itself doesn't depend
        on which gene(s) are being scored).

        Args:
            target_class (int or str): label for the class of interest
            n_tests (int): number of permutations
            alpha (float): significance threshold for multiple test correction
            multiple_test_method (str): correction method from statsmodels.stats.multitest.multipletests

        Returns:
            pd.DataFrame: one row per gene, columns:
                - signed score   : mean(SHAP[Case]) - mean(SHAP[Control]) for that gene
                - direction      : "toward_<class>" if signed > 0 else "away_from_<class>"
                - p_pos corrected: FDR-corrected P(null >= signed_score) [enriched toward target class]
                - p_neg corrected: FDR-corrected P(null <= signed_score) [enriched away from target class]
              Call a gene significant when min(p_pos corrected, p_neg corrected) < alpha;
              `direction` says which one to look at. Same schema as explain_gene_sets, just
              indexed by gene instead of gene set.

        Note:
            The null array built here is (n_tests, n_genes) - fine for the gene panels used in
            practice (a filtered pathway-library gene union, a few hundred to low thousands of
            genes), but not intended for an unfiltered whole transcriptome at very large n_tests.
        """
        global_gene_enrichment_score = self.compute_target_global_gene_enrichment_score(target_class)

        # Seed once, same as explain_gene_sets, so the whole call is reproducible under
        # self.random_state.
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

        # One shared set of n_tests permutations, reused across every gene: for each permutation,
        # compute the Case-mean - Control-mean contrast for ALL genes at once via a vectorized
        # mean over the SHAP matrix, instead of drawing a fresh permutation per gene per test.
        null = np.empty((n_tests, shap_matrix.shape[1]))
        for i in range(n_tests):
            perm = self.pseudo_random_generator.permutation(n_cells)
            null[i] = shap_matrix[perm[:n_case]].mean(axis=0) - shap_matrix[perm[n_case:]].mean(axis=0)

        signed_scores = global_gene_enrichment_score
        p_pos = pd.Series((null >= signed_scores.values[None, :]).mean(axis=0), index=signed_scores.index)
        p_neg = pd.Series((null <= signed_scores.values[None, :]).mean(axis=0), index=signed_scores.index)

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
        self.gene_importance["Is Selected"] = (outliers == -1)
        return self.gene_importance

    # ------------------------------------------------------------------------------------
    # Plotting
    #
    # Fixed diverging color roles, used consistently across all three plots: "toward" the
    # target class is always blue, "away from" it is always red, "not significant" is always
    # neutral gray. Matches the direction/polarity semantics that explain_genes and
    # explain_gene_sets already use.
    # ------------------------------------------------------------------------------------
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
        Two-sided combined p-value from a p_pos/p_neg-corrected result table (2*min(p_pos, p_neg)
        corrected, clipped to 1).
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
        import matplotlib.pyplot as plt

        if res is None:
            res = getattr(self, "gene_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_genes(...) first, or pass res=.")

        pvals = self._combined_pvalue(res)
        # A permutation test can only resolve p down to ~1/n_tests; an exact-zero p-value means
        # "smaller than that resolution", not "~1e-300". Clipping at float epsilon would claim
        # false precision and blow the y-axis out to ~300, squashing every other point against
        # the bottom. Floor zeros at one order of magnitude below min(alpha, smallest observed
        # nonzero p) instead - always below alpha, so a point that IS significant never plots
        # below the alpha reference line, which floor=nonzero_min/10 alone can't guarantee (e.g.
        # if every nonzero p in the data happens to sit above alpha already).
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

        # Stagger labels deterministically by rank so near-coincident points (common with
        # permutation p-values, which take on few distinct values) don't render as illegible
        # overlapping text - each successive label steps further out and alternates side.
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
        Dot plot of the top_n gene sets from explain_gene_sets(), ranked by |signed score|
        (the order explain_gene_sets already returns them in): one row per gene set, dot color
        by direction, dot size by significance (larger = smaller combined p-value).

        Args:
            res (pd.DataFrame): output of explain_gene_sets(). Defaults to the result of the
                most recent explain_gene_sets(...) call on this instance.
            top_n (int): number of top gene sets to show.
            ax (matplotlib.axes.Axes): axes to draw on; a new figure is created if None.

        Returns:
            matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if res is None:
            res = getattr(self, "gene_set_results_", None)
            if res is None:
                raise ValueError("No results to plot. Call explain_gene_sets(...) first, or pass res=.")

        # smallest-to-largest so the single strongest hit ends up at the top of the plot
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
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

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
        expression z-scored across the explained cells (not SHAP values) - the standard
        "top DE genes" heatmap that typically accompanies a differential expression analysis.

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
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.lines import Line2D

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

        # z-score each gene (row) across the explained cells, for visual contrast
        mu = Z.mean(axis=1, keepdims=True)
        sigma = Z.std(axis=1, keepdims=True)
        sigma[sigma == 0] = 1.0
        Z = (Z - mu) / sigma

        # Diverging blue-low/red-high colormap built from the same hues used elsewhere in
        # this class, for a consistent palette across all four plot types - though here
        # color encodes "expression relative to this gene's own mean", not direction of
        # significance, so it is labeled separately via the colorbar rather than reusing
        # the toward/away legend.
        cmap = LinearSegmentedColormap.from_list(
            "gshapa_diverging", [self._PLOT_TOWARD_COLOR, "#f0efec", self._PLOT_AWAY_COLOR]
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
            strip_colors = ["#eb6834", self._PLOT_MUTED]  # categorical slot 2 (orange) vs muted grey
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
