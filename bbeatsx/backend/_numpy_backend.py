"""Pure-numpy reference backend mirroring the ``stochtree`` low-level API.

BBEATSx is written **once** against the ``stochtree`` low-level primitives
(``Dataset``, ``Residual``, ``Forest``, ``ForestContainer``, ``ForestSampler``,
``GlobalVarianceModel``, ``LeafVarianceModel``, ``ForestModelConfig``,
``GlobalModelConfig``, ``RNG``).  When the compiled ``stochtree_cpp`` extension is
unavailable, :mod:`bbeatsx.backend` falls back to the classes defined here so the
package remains importable, runnable, and testable.

This is a *reference* implementation: a compact-but-genuine Bayesian backfitting
BART with single grow/prune Metropolis-Hastings moves per tree and conjugate
Gaussian leaves.  It reproduces the semantics BBEATSx depends on -- a shared,
continuously-updated residual; per-observation precision weights with the
``y_i ~ N(mu, sigma^2 / w_i)`` convention; piecewise-constant (extrapolation-
flatlining) predictions -- but it is **not** performance-optimized and is not a
drop-in replacement for the production C++ sampler.

Weight convention matches stochtree: ``variance_weights`` are *precision*
multipliers, i.e. ``Var(eps_i) = sigma^2 / w_i``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np

BACKEND_NAME = "numpy-reference"


# --------------------------------------------------------------------------- RNG
class RNG:
    """Wrapper around :class:`numpy.random.Generator` (mirrors ``stochtree.RNG``)."""

    def __init__(self, random_seed: int = -1) -> None:
        seed = None if random_seed is None or random_seed < 0 else int(random_seed)
        self.rng = np.random.default_rng(seed)


# ----------------------------------------------------------------------- Dataset
class Dataset:
    """Stores covariates and optional per-observation precision weights."""

    def __init__(self) -> None:
        self._X: Optional[np.ndarray] = None
        self._w: Optional[np.ndarray] = None

    def add_covariates(self, covariates: np.ndarray) -> None:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        self._X = np.ascontiguousarray(X)

    def add_variance_weights(self, variance_weights: np.ndarray) -> None:
        self._w = np.asarray(variance_weights, dtype=float).ravel().copy()

    def update_variance_weights(self, variance_weights: np.ndarray,
                                exponentiate: bool = False) -> None:
        w = np.asarray(variance_weights, dtype=float).ravel()
        self._w = np.exp(w) if exponentiate else w.copy()

    def num_observations(self) -> int:
        return 0 if self._X is None else self._X.shape[0]

    def num_covariates(self) -> int:
        return 0 if self._X is None else self._X.shape[1]

    def get_covariates(self) -> np.ndarray:
        return self._X

    def has_variance_weights(self) -> bool:
        return self._w is not None

    def get_variance_weights(self) -> Optional[np.ndarray]:
        return self._w


# ---------------------------------------------------------------------- Residual
class Residual:
    """Continuously-updated (full or partial) residual stream."""

    def __init__(self, residual: np.ndarray) -> None:
        self._data = np.asarray(residual, dtype=float).ravel().copy()

    def get_residual(self) -> np.ndarray:
        return self._data.copy()

    def update_data(self, new_vector: np.ndarray) -> None:
        self._data = np.asarray(new_vector, dtype=float).ravel().copy()

    def add_vector(self, update_vector: np.ndarray) -> None:
        self._data += np.asarray(update_vector, dtype=float).ravel()

    def subtract_vector(self, update_vector: np.ndarray) -> None:
        self._data -= np.asarray(update_vector, dtype=float).ravel()


# ------------------------------------------------------------------ model config
class GlobalModelConfig:
    def __init__(self, global_error_variance: float = 1.0) -> None:
        self.global_error_variance = float(global_error_variance)

    def update_global_error_variance(self, v: float) -> None:
        self.global_error_variance = float(v)

    def get_global_error_variance(self) -> float:
        return self.global_error_variance


class ForestModelConfig:
    """Subset of ``stochtree.ForestModelConfig`` used by BBEATSx."""

    def __init__(
        self,
        num_trees: int = None,
        num_features: int = None,
        num_observations: int = None,
        feature_types=None,
        variable_weights=None,
        leaf_dimension: int = 1,
        alpha: float = 0.95,
        beta: float = 2.0,
        min_samples_leaf: int = 5,
        max_depth: int = -1,
        leaf_model_type: int = 0,
        leaf_model_scale: Union[float, np.ndarray, None] = None,
        cutpoint_grid_size: int = 100,
        num_features_subsample=None,
        **_ignored,
    ) -> None:
        if num_trees is None:
            raise ValueError("`num_trees` must be provided")
        if num_features is None and feature_types is not None:
            num_features = len(feature_types)
        if num_features is None:
            raise ValueError("`num_features` or `feature_types` must be provided")
        self.num_trees = int(num_trees)
        self.num_features = int(num_features)
        self.num_observations = num_observations
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = int(max_depth)
        self.leaf_model_type = int(leaf_model_type)
        self.cutpoint_grid_size = int(cutpoint_grid_size)
        if variable_weights is None:
            self.variable_weights = np.full(self.num_features,
                                            1.0 / max(self.num_features, 1))
        else:
            self.variable_weights = np.asarray(variable_weights, dtype=float)
        if leaf_model_scale is None:
            self._leaf_scale = np.array([[1.0 / self.num_trees]])
        elif isinstance(leaf_model_scale, np.ndarray):
            self._leaf_scale = leaf_model_scale
        else:
            self._leaf_scale = np.array([[float(leaf_model_scale)]])
        if feature_types is None:
            self.feature_types = np.zeros(self.num_features, dtype=int)
        else:
            self.feature_types = np.asarray(feature_types, dtype=int)

    # getters used by the sampler
    def get_alpha(self): return self.alpha
    def get_beta(self): return self.beta
    def get_min_samples_leaf(self): return self.min_samples_leaf
    def get_max_depth(self): return self.max_depth
    def get_num_trees(self): return self.num_trees
    def get_num_features(self): return self.num_features
    def get_num_observations(self): return self.num_observations
    def get_variable_weights(self): return self.variable_weights
    def get_leaf_model_scale(self): return self._leaf_scale
    def get_leaf_model_type(self): return self.leaf_model_type
    def get_feature_types(self): return self.feature_types

    def update_leaf_model_scale(self, leaf_model_scale) -> None:
        if isinstance(leaf_model_scale, np.ndarray):
            self._leaf_scale = leaf_model_scale
        else:
            self._leaf_scale = np.array([[float(leaf_model_scale)]])

    def update_beta(self, beta: float) -> None:
        self.beta = float(beta)


# ------------------------------------------------------------------------- trees
class _Node:
    __slots__ = ("leaf", "depth", "idx", "feat", "thresh", "value",
                 "left", "right", "parent")

    def __init__(self, depth: int, idx: np.ndarray, value: float = 0.0,
                 parent: "Optional[_Node]" = None) -> None:
        self.leaf = True
        self.depth = depth
        self.idx = idx
        self.feat: Optional[int] = None
        self.thresh: Optional[float] = None
        self.value = value
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None
        self.parent = parent


def _leaves(node: _Node) -> List[_Node]:
    out: List[_Node] = []
    stack = [node]
    while stack:
        nd = stack.pop()
        if nd.leaf:
            out.append(nd)
        else:
            stack.append(nd.left)
            stack.append(nd.right)
    return out


def _pruneable(node: _Node) -> List[_Node]:
    """Internal nodes both of whose children are leaves."""
    out: List[_Node] = []
    stack = [node]
    while stack:
        nd = stack.pop()
        if not nd.leaf:
            if nd.left.leaf and nd.right.leaf:
                out.append(nd)
            else:
                stack.append(nd.left)
                stack.append(nd.right)
    return out


def _split_count(node: _Node, counts: np.ndarray) -> None:
    if not node.leaf:
        counts[node.feat] += 1
        _split_count(node.left, counts)
        _split_count(node.right, counts)


class _Tree:
    """A single regression tree with constant leaves."""

    def __init__(self, root_value: float = 0.0) -> None:
        self.root = _Node(depth=0, idx=None, value=root_value)

    # ---- prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=float)
        stack = [(self.root, np.arange(X.shape[0]))]
        while stack:
            nd, idx = stack.pop()
            if nd.leaf:
                out[idx] = nd.value
            else:
                go_left = X[idx, nd.feat] <= nd.thresh
                stack.append((nd.left, idx[go_left]))
                stack.append((nd.right, idx[~go_left]))
        return out

    def split_counts(self, num_features: int) -> np.ndarray:
        counts = np.zeros(num_features, dtype=float)
        _split_count(self.root, counts)
        return counts

    # ---- structural clone (strip training indices) for snapshots
    def clone_structure(self) -> "_Tree":
        new = _Tree.__new__(_Tree)

        def _clone(nd: _Node) -> _Node:
            c = _Node(nd.depth, None, nd.value)
            c.leaf = nd.leaf
            c.feat = nd.feat
            c.thresh = nd.thresh
            if not nd.leaf:
                c.left = _clone(nd.left)
                c.right = _clone(nd.right)
            return c

        new.root = _clone(self.root)
        return new


# ---------------------------------------------------------------------- forests
class Forest:
    """The "active" tree ensemble being sampled (constant-leaf model)."""

    def __init__(self, num_trees: int, output_dimension: int = 1,
                 leaf_constant: bool = True, is_exponentiated: bool = False) -> None:
        self.num_trees = int(num_trees)
        self._trees: List[_Tree] = [_Tree() for _ in range(self.num_trees)]
        self.internal_forest_is_empty = True

    def is_empty(self) -> bool:
        return self.internal_forest_is_empty

    def set_root_leaves(self, leaf_value) -> None:
        val = float(np.squeeze(leaf_value)) if isinstance(leaf_value, np.ndarray) \
            else float(leaf_value)
        for tr in self._trees:
            tr.root = _Node(depth=0, idx=None, value=val)
        self.internal_forest_is_empty = False

    def predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.get_covariates()
        total = np.zeros(X.shape[0], dtype=float)
        for tr in self._trees:
            total += tr.predict(X)
        return total

    def total_leaf_sq(self) -> "tuple[float, int]":
        s, c = 0.0, 0
        for tr in self._trees:
            for lf in _leaves(tr.root):
                s += lf.value ** 2
                c += 1
        return s, c


class ForestContainer:
    """Stores forest snapshots retained across MCMC draws."""

    def __init__(self, num_trees: int, output_dimension: int = 1,
                 leaf_constant: bool = True, is_exponentiated: bool = False) -> None:
        self.num_trees = int(num_trees)
        self._forests: List[List[_Tree]] = []

    def append_snapshot(self, trees: List[_Tree]) -> None:
        self._forests.append([t.clone_structure() for t in trees])

    def num_samples(self) -> int:
        return len(self._forests)

    def predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.get_covariates()
        n = X.shape[0]
        S = len(self._forests)
        out = np.zeros((n, S), dtype=float)
        for s, trees in enumerate(self._forests):
            acc = np.zeros(n, dtype=float)
            for tr in trees:
                acc += tr.predict(X)
            out[:, s] = acc
        return out

    def predict_raw_single_forest(self, dataset: Dataset, forest_num: int) -> np.ndarray:
        X = dataset.get_covariates()
        acc = np.zeros(X.shape[0], dtype=float)
        for tr in self._forests[forest_num]:
            acc += tr.predict(X)
        return acc

    def get_forest_split_counts(self, forest_num: int, num_features: int) -> np.ndarray:
        counts = np.zeros(num_features, dtype=float)
        for tr in self._forests[forest_num]:
            counts += tr.split_counts(num_features)
        return counts

    def get_overall_split_counts(self, num_features: int) -> np.ndarray:
        counts = np.zeros(num_features, dtype=float)
        for trees in self._forests:
            for tr in trees:
                counts += tr.split_counts(num_features)
        return counts


# --------------------------------------------------------------- forest sampler
class ForestSampler:
    """Single grow/prune Bayesian backfitting sampler over a shared residual."""

    def __init__(self, dataset: Dataset, global_config: GlobalModelConfig,
                 forest_config: ForestModelConfig) -> None:
        self.dataset = dataset
        self.num_trees = forest_config.get_num_trees()

    # ---- initialization
    def prepare_for_sampler(self, dataset: Dataset, residual: Residual,
                            forest: Forest, leaf_model: int,
                            initial_values) -> None:
        init = float(np.squeeze(initial_values)) if isinstance(initial_values,
                                                               np.ndarray) \
            else float(initial_values)
        n = dataset.num_observations()
        per_tree = init / forest.num_trees
        all_idx = np.arange(n)
        for tr in forest._trees:
            tr.root = _Node(depth=0, idx=all_idx, value=per_tree)
        forest.internal_forest_is_empty = False
        # Subtract the (constant) forest prediction from the residual.
        residual.subtract_vector(forest.predict(dataset))

    # ---- one Gibbs sweep over the forest
    def sample_one_iteration(self, forest_container: ForestContainer, forest: Forest,
                             dataset: Dataset, residual: Residual, rng: RNG,
                             global_config: GlobalModelConfig,
                             forest_config: ForestModelConfig,
                             keep_forest: bool, gfr: bool,
                             num_threads: int = 1) -> None:
        if forest.is_empty():
            raise ValueError("forest not initialized; call prepare_for_sampler first")
        X = dataset.get_covariates()
        w = dataset.get_variance_weights()
        sigma2 = global_config.get_global_error_variance()
        prec = (np.ones(X.shape[0]) if w is None else w) / sigma2
        tau = float(forest_config.get_leaf_model_scale()[0, 0])
        alpha = forest_config.get_alpha()
        beta = forest_config.get_beta()
        msl = forest_config.get_min_samples_leaf()
        max_depth = forest_config.get_max_depth()
        n_features = forest_config.get_num_features()
        gen = rng.rng

        n_struct_moves = 3 if gfr else 1
        for tr in forest._trees:
            # 1. add this tree's current prediction back -> partial residual R_t.
            old_pred = tr.predict(X)
            residual.add_vector(old_pred)
            r = residual.get_residual()
            # 2. structure move(s).
            for _ in range(n_struct_moves):
                self._structure_move(tr, X, r, prec, tau, alpha, beta, msl,
                                     max_depth, n_features, gen, force_grow=gfr)
            # 3. redraw all leaf values from the Gaussian conjugate posterior.
            self._draw_leaves(tr, r, prec, tau, gen)
            # 4. subtract the new prediction.
            residual.subtract_vector(tr.predict(X))
        if keep_forest:
            forest_container.append_snapshot(forest._trees)

    # ---- conjugate leaf-marginal log likelihood for a set of rows
    @staticmethod
    def _lml(idx: np.ndarray, r: np.ndarray, prec: np.ndarray, a0: float) -> float:
        P = prec[idx].sum()
        Q = (prec[idx] * r[idx]).sum()
        return 0.5 * math.log(a0 / (a0 + P)) + 0.5 * (Q * Q) / (a0 + P)

    def _structure_move(self, tree: _Tree, X, r, prec, tau, alpha, beta, msl,
                        max_depth, n_features, gen, force_grow: bool) -> None:
        a0 = 1.0 / tau
        leaves = _leaves(tree.root)
        single_root = tree.root.leaf
        # choose move
        if single_root:
            do_grow = True
        elif force_grow:
            do_grow = True
        else:
            do_grow = gen.random() < 0.5

        if do_grow:
            self._try_grow(tree, leaves, X, r, prec, a0, alpha, beta, msl,
                           max_depth, n_features, gen, single_root)
        else:
            self._try_prune(tree, X, r, prec, a0, alpha, beta, gen)

    def _try_grow(self, tree, leaves, X, r, prec, a0, alpha, beta, msl,
                  max_depth, n_features, gen, single_root) -> None:
        node = leaves[gen.integers(len(leaves))]
        d = node.depth
        if max_depth >= 0 and d + 1 > max_depth:
            return
        idx = node.idx
        if idx.shape[0] < 2 * msl:
            return
        # available features with >=1 valid cutpoint
        feats = np.arange(n_features)
        gen.shuffle(feats)
        chosen = None
        for f in feats:
            vals = X[idx, f]
            uniq = np.unique(vals)
            if uniq.shape[0] < 2:
                continue
            # candidate thresholds = unique values except the maximum
            cands = uniq[:-1]
            gen.shuffle(cands)
            for v in cands:
                left = idx[X[idx, f] <= v]
                nl = left.shape[0]
                nr = idx.shape[0] - nl
                if nl >= msl and nr >= msl:
                    chosen = (int(f), float(v), left, idx[X[idx, f] > v])
                    break
            if chosen is not None:
                break
        if chosen is None:
            return
        f, v, left_idx, right_idx = chosen

        lml_parent = self._lml(idx, r, prec, a0)
        lml_left = self._lml(left_idx, r, prec, a0)
        lml_right = self._lml(right_idx, r, prec, a0)
        lml_ratio = lml_left + lml_right - lml_parent

        ps_d = alpha * (1.0 + d) ** (-beta)
        ps_d1 = alpha * (2.0 + d) ** (-beta)
        log_prior = (math.log(ps_d) + 2.0 * math.log(1.0 - ps_d1)
                     - math.log(1.0 - ps_d))

        b = len(leaves)
        w2_star = len(_pruneable(tree.root))
        # growing `node`: its parent (if previously pruneable) stops being
        # pruneable; the new split node becomes pruneable.
        if node.parent is not None and node.parent.left.leaf and node.parent.right.leaf:
            w2_star_new = w2_star - 1 + 1
        else:
            w2_star_new = w2_star + 1
        p_grow = 1.0 if single_root else 0.5
        p_prune_rev = 0.5  # grown tree is never single-root
        log_trans = math.log(p_prune_rev / p_grow) + math.log(b) - math.log(w2_star_new)

        log_accept = lml_ratio + log_prior + log_trans
        if math.log(gen.random() + 1e-300) < log_accept:
            node.leaf = False
            node.feat = f
            node.thresh = v
            node.left = _Node(d + 1, left_idx, parent=node)
            node.right = _Node(d + 1, right_idx, parent=node)

    def _try_prune(self, tree, X, r, prec, a0, alpha, beta, gen) -> None:
        prune_nodes = _pruneable(tree.root)
        if not prune_nodes:
            return
        node = prune_nodes[gen.integers(len(prune_nodes))]
        d = node.depth
        idx = node.idx
        left_idx = node.left.idx
        right_idx = node.right.idx

        lml_parent = self._lml(idx, r, prec, a0)
        lml_left = self._lml(left_idx, r, prec, a0)
        lml_right = self._lml(right_idx, r, prec, a0)
        # ratio for *prune* = inverse of grow ratio.
        lml_ratio = lml_parent - (lml_left + lml_right)

        ps_d = alpha * (1.0 + d) ** (-beta)
        ps_d1 = alpha * (2.0 + d) ** (-beta)
        log_prior = (math.log(1.0 - ps_d)
                     - (math.log(ps_d) + 2.0 * math.log(1.0 - ps_d1)))

        b = len(_leaves(tree.root))            # leaves before prune
        w2 = len(prune_nodes)                  # pruneable before prune
        # after prune: leaves b-1; the merged node may become pruneable for parent
        pruned_single_root = (node.parent is None)
        p_prune = 0.5
        p_grow_rev = 1.0 if pruned_single_root else 0.5
        log_trans = math.log(p_grow_rev / p_prune) + math.log(w2) - math.log(b - 1)

        log_accept = lml_ratio + log_prior + log_trans
        if math.log(gen.random() + 1e-300) < log_accept:
            node.leaf = True
            node.feat = None
            node.thresh = None
            node.left = None
            node.right = None

    @staticmethod
    def _draw_leaves(tree: _Tree, r, prec, tau, gen) -> None:
        a0 = 1.0 / tau
        for lf in _leaves(tree.root):
            idx = lf.idx
            if idx is None or idx.shape[0] == 0:
                lf.value = gen.normal(0.0, math.sqrt(tau))
                continue
            P = prec[idx].sum()
            Q = (prec[idx] * r[idx]).sum()
            post_prec = a0 + P
            mean = Q / post_prec
            lf.value = gen.normal(mean, math.sqrt(1.0 / post_prec))


# ------------------------------------------------------------ variance samplers
class GlobalVarianceModel:
    """Inverse-gamma global error variance draw (homoscedastic path)."""

    def sample_one_iteration(self, residual: Residual, rng: RNG,
                             a: float, b: float) -> float:
        r = residual.get_residual()
        n = r.shape[0]
        a_post = a + 0.5 * n
        b_post = b + 0.5 * float(np.sum(r * r))
        return float(b_post / rng.rng.standard_gamma(a_post))


class LeafVarianceModel:
    """Inverse-gamma leaf-scale (sigma_mu^2) draw for one forest."""

    def sample_one_iteration(self, forest: Forest, rng: RNG,
                             a: float, b: float) -> float:
        s, c = forest.total_leaf_sq()
        a_post = a + 0.5 * c
        b_post = b + 0.5 * s
        return float(b_post / rng.rng.standard_gamma(a_post))
