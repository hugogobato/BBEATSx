"""Configuration dataclasses for BBEATSx (Phase 1, plan §1.5).

Every prior, toggle, and sampler setting flows through these dataclasses so the
model variants enumerated in the research plan -- ``trend in {tree, linear,
spline, tvp}``, ``errors in {homo, sv}``, asymmetric component priors on/off,
recursive vs direct multi-step, seasonal sum-to-zero on/off -- are all
switchable from a single config object (Phase 1 exit criterion).

The defaults follow the plan's recommendations: ``trend="spline"``, homoscedastic
errors by default (SV opt-in, intended to be chosen by LOO/marginal likelihood),
recursive multi-step forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

TrendMode = Literal["tree", "linear", "spline", "tvp"]
ErrorMode = Literal["homo", "sv"]
MultiStepMode = Literal["recursive", "direct"]


@dataclass
class SeasonalPeriod:
    """A single seasonal frequency expressed as Fourier harmonics.

    Parameters
    ----------
    period : float
        Length of one seasonal cycle in time-index units (e.g. 7 for weekly
        seasonality on daily data, 24 for daily on hourly data, 12 for yearly
        on monthly data).
    harmonics : int
        Number of sin/cos harmonic pairs ``k = 1..harmonics`` to generate.
    """

    period: float
    harmonics: int = 3

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("SeasonalPeriod.period must be positive")
        if self.harmonics < 1:
            raise ValueError("SeasonalPeriod.harmonics must be >= 1")


@dataclass
class TreePrior:
    """BART tree-structure + leaf priors for one forest component.

    The depth-penalty ``beta`` and leaf scale ``leaf_scale`` are the levers the
    plan uses to make the *generic* block shallow and tightly regularized for
    identifiability (concept §3.5 / plan §1.2).

    Parameters
    ----------
    num_trees : int
        Number of trees in the ensemble (``m_c``).
    alpha, beta : float
        Tree-structure prior ``p(split at depth d) = alpha * (1 + d) ** -beta``.
    leaf_scale : float, optional
        Prior leaf standard deviation ``sigma_mu`` *on the standardized scale*.
        If ``None`` it is calibrated to ``leaf_scale_factor / sqrt(num_trees)``
        (the standard BART half-range heuristic on unit-variance data).
    leaf_scale_factor : float
        Calibration constant used when ``leaf_scale is None``.
    min_samples_leaf : int
        Minimum training observations per leaf.
    max_depth : int
        Maximum tree depth (``-1`` = unbounded).
    sample_leaf_scale : bool
        If ``True``, place a per-component ``LeafVarianceModel`` draw on
        ``sigma_mu^2`` (plan §1.2 step 5).
    """

    num_trees: int = 50
    alpha: float = 0.95
    beta: float = 2.0
    leaf_scale: Optional[float] = None
    leaf_scale_factor: float = 3.0
    min_samples_leaf: int = 5
    max_depth: int = -1
    sample_leaf_scale: bool = False

    def resolved_leaf_scale(self) -> float:
        """Return ``sigma_mu`` on the standardized scale, calibrating if needed."""
        if self.leaf_scale is not None:
            return float(self.leaf_scale)
        # Half the plausible standardized range (~3 sd) spread over sqrt(m) trees.
        return self.leaf_scale_factor / (2.0 * (self.num_trees ** 0.5))


@dataclass
class TrendConfig:
    """Trend block configuration (plan §1.2 step 1, §3.6).

    ``mode`` selects the extrapolation behaviour:

    - ``"linear"`` / ``"spline"`` (**default**): conjugate Gaussian basis
      regression ``F_tr(t) = phi(t) @ beta`` -- extrapolation-safe (Lemma 2.3).
    - ``"tvp"``: time-varying coefficients ``beta_{k} = BART_k(z_tvp)`` with the
      model staying linear in ``phi(t)`` (the TVP-BART-inspired variant).
    - ``"tree"``: a BART forest on engineered ``t``-features -- kept purely as
      the extrapolation-failure ablation/foil.
    """

    mode: TrendMode = "spline"
    # Linear/poly degree (used when mode == "linear").
    degree: int = 1
    # B-spline settings (used when mode == "spline").
    n_knots: int = 10
    spline_degree: int = 3
    # Gaussian prior sd on basis coefficients (ridge); standardized scale.
    coef_scale: float = 10.0
    # Random-walk smoothing prior strength for P-spline (penalizes 2nd diffs).
    smoothing: float = 1.0
    # Forest prior used only when mode in {"tree", "tvp"}.
    tree_prior: TreePrior = field(
        default_factory=lambda: TreePrior(num_trees=50, beta=2.0)
    )


@dataclass
class SeasonalConfig:
    """Seasonality block configuration (plan §1.2 step 2)."""

    periods: List[SeasonalPeriod] = field(default_factory=list)
    # Optional calendar one-hot columns to include (resolved by FeatureBuilder).
    calendar: List[str] = field(default_factory=list)
    # Sum-to-zero centering of the seasonal component per sweep (concept §3.5).
    sum_to_zero: bool = True
    tree_prior: TreePrior = field(
        default_factory=lambda: TreePrior(num_trees=50, beta=2.0)
    )


@dataclass
class GenericConfig:
    """Generic block configuration: AR lags + exogenous covariates (§1.2 step 3).

    ``asymmetric=True`` activates the identifiability prior (concept §3.5): fewer
    trees, a tighter leaf scale and a steeper depth penalty so the block makes
    only small corrections and does not steal trend/seasonality.
    """

    lags: Sequence[int] = (1,)
    exog: List[str] = field(default_factory=list)
    # Exogenous covariates that are known into the future (the "-x" features).
    future_exog: List[str] = field(default_factory=list)
    asymmetric: bool = True
    tree_prior: TreePrior = field(
        default_factory=lambda: TreePrior(num_trees=50, beta=2.0)
    )

    def resolved_tree_prior(self) -> TreePrior:
        """Return the (possibly asymmetric-tightened) tree prior for this block."""
        tp = self.tree_prior
        if not self.asymmetric:
            return tp
        # Asymmetric prior: fewer trees, steeper depth penalty, tighter leaves.
        return TreePrior(
            num_trees=max(10, tp.num_trees // 2),
            alpha=tp.alpha,
            beta=tp.beta + 1.0,
            leaf_scale=(tp.leaf_scale if tp.leaf_scale is not None
                        else (tp.leaf_scale_factor / (4.0 * (tp.num_trees ** 0.5)))),
            leaf_scale_factor=tp.leaf_scale_factor,
            min_samples_leaf=tp.min_samples_leaf,
            max_depth=tp.max_depth,
            sample_leaf_scale=tp.sample_leaf_scale,
        )


@dataclass
class ErrorConfig:
    """Observation-error model (plan §0.2, §1.2 step 4).

    ``mode="homo"`` -> homoscedastic ``N(0, sigma^2)`` with an inverse-gamma
    ``GlobalVarianceModel`` draw. ``mode="sv"`` -> log-variance follows an AR(1),
    sampled with the Omori et al. (2007) 10-component mixture.
    """

    mode: ErrorMode = "homo"
    # Inverse-gamma prior on sigma^2 (homoscedastic): nu/2, nu*lambda/2.
    nu: float = 3.0
    q: float = 0.90  # quantile used to calibrate lambda from the data
    # SV AR(1) prior hyperparameters: h_t = mu + phi (h_{t-1} - mu) + sigma_h nu_t.
    sv_phi: float = 0.95
    sv_sigma_h: float = 0.15
    sv_mu_prior_var: float = 10.0


@dataclass
class MCMCConfig:
    """Sampler schedule and reproducibility settings."""

    num_gfr: int = 10          # grow-from-root warm-start sweeps (XBART-style init)
    num_burnin: int = 200
    num_mcmc: int = 500
    thin: int = 1
    seed: int = 0
    num_threads: int = 1


@dataclass
class BBEATSxConfig:
    """Top-level BBEATSx configuration bundling every block and toggle."""

    trend: TrendConfig = field(default_factory=TrendConfig)
    seasonal: SeasonalConfig = field(default_factory=SeasonalConfig)
    generic: GenericConfig = field(default_factory=GenericConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    multistep: MultiStepMode = "recursive"

    def num_retained_draws(self) -> int:
        """Number of posterior draws kept after burn-in and thinning."""
        return self.mcmc.num_mcmc // self.mcmc.thin
