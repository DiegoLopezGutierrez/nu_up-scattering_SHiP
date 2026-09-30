"""
Plot the HNL production flux at the SHiP proton target from the output of
../geant4/target_sim (see ../geant4/README.md for the simulation itself).

The Geant4 ntuple stores, per candidate meson-decay channel and HNL
benchmark mass, a mixing-independent weight

    weightPerU2 = BR(h -> l_alpha N) / |U_alpha|^2

so that the physical flux for any active-sterile mixing pattern
(Ue2, Umu2, Utau2) can be obtained here -- without re-running the
(expensive) Geant4/Pythia8 simulation -- by summing

    weight = weightPerU2 * U_alpha^2

over all produced mesons, normalized per simulated proton (POT).

Uses uproot (not PyROOT/ROOT.TFile as elsewhere in this repo) since PyROOT
was found to be broken in this environment (ROOT 6.36 + cppyy crashes on
basic TTree access); uproot reads the same G4Analysis-written ROOT file
without needing a working ROOT/Cling installation.
"""

import argparse
from zlib import Z_DEFAULT_COMPRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import integrate
import uproot

STYLE_DIR = "../plots/"
plt.style.use(STYLE_DIR + "sty.mplstyle")


def _log_tick_formatter(value, _pos=None):
    """
    Format log-axis decade ticks as "$10^{n}$". Only exact decades are
    labeled, matching the default matplotlib log-tick behavior (minor
    ticks between decades are left unlabeled).
    """
    if value <= 0:
        return ""
    exponent = np.log10(value)
    if not np.isclose(exponent, round(exponent), atol=1e-6):
        return ""
    return rf"$10^{{{int(round(exponent))}}}$"


def set_log_ticks(ax) -> None:
    """
    Apply the decade-only "$10^n$" formatter to both axes, and render the
    resulting tick labels with matplotlib's own mathtext engine instead of
    real LaTeX.

    sty.mplstyle sets text.usetex=True, but matplotlib's PDF backend has a
    real bug where the minus sign is silently dropped from usetex-rendered
    math text -- confirmed independent of this formatter (a bare "$-8$"
    reproduces it) and independent of matplotlib's own default formatter
    (which wraps labels in a "\\mathdefault{}" macro); it also only
    affects the PDF backend, not PNG (both shell out to the same LaTeX).
    So e.g. "$10^{-6}$" renders as a bare "10" with the "-6" silently
    dropped, but only in the saved PDF. Forcing usetex off just for the
    tick labels renders them via mathtext (immune to this bug) while
    leaving the rest of the figure (titles, axis labels, legend) on real
    LaTeX/Palatino as before -- a minor, localized font mismatch traded
    for actually-correct exponents.
    """
    ax.xaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))

    ax.figure.canvas.draw()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_usetex(False)

LEPTON_LABEL = {11: r"$e$", 13: r"$\mu$", 15: r"$\tau$"}
LEPTON_NAME = {11: "e", 13: "mu", 15: "tau"}  # plain-text labels, for the text summary

PARENT_LABEL = {
    211: r"$\pi^\pm$", 321: r"$K^\pm$",
    411: r"$D^\pm$", 431: r"$D_s^\pm$",
    521: r"$B^\pm$", 541: r"$B_c^\pm$",
}

# One color per meson species, for --breakdown plots.
PARENT_COLORS = {
    211: "#0C5DA5", 321: "#00B945",
    411: "#FF9500", 431: "#FF2C00",
    521: "#845B97", 541: "#474747",
}

LIGHT_PARENTS = (211, 321)
HEAVY_PARENTS = (411, 431, 521, 541)

# SHiP decay volume geometry, all in mm.
Z_DECAY_VOLUME = 33.5e3  # distance from the target (z=0) to the decay volume
X_DECAY_VOLUME = 0.5e3   # half width
Y_DECAY_VOLUME = 1.35e3  # half height

G_F = 1.1663787e-5  # Fermi constant [GeV^-2]

# Charged-lepton masses [GeV] (PDG), keyed by PDG code.
LEPTON_MASS = {11: 0.000511, 13: 0.105658, 15: 1.77686}


def _kallen(a: float, b: float, c: float) -> float:
    """Kallen (triangle) function lambda(a,b,c), Bondarenko et al. eq. (2.10)."""
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a


def _cc_phase_space_integral(x_u: float, x_d: float, x_l: float) -> float:
    """
    Phase-space integral I(x_u, x_d, x_l) of Bondarenko et al. (arXiv:1805.08567)
    eq. (3.2), the finite-mass correction factor for the charged-current
    3-body decay N -> l U Dbar.
    """
    lower = (x_d + x_l) ** 2
    upper = (1.0 - x_u) ** 2
    if lower >= upper:
        return 0.0  # kinematically closed

    def integrand(x):
        rad = max(_kallen(x, x_l**2, x_d**2) * _kallen(1.0, x, x_u**2), 0.0)
        return (x - x_l**2 - x_d**2) * (1.0 + x_u**2 - x) * np.sqrt(rad) / x

    value, _ = integrate.quad(integrand, lower, upper)
    return 12.0 * value


def cc_leptonic_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Charged-current-mediated 3-body leptonic HNL decay width [GeV],
    Gamma(N -> l_alpha^- nu_beta l_beta^+) summed over alpha != beta and
    over all kinematically open (alpha, beta) pairs, weighted by the
    active-sterile mixing |U_alpha|^2. From Bondarenko et al.
    (arXiv:1805.08567) Section 3.1.1, eqs. (3.1)-(3.2), with N_W = 1.

    This is a *partial* decay width: it omits (a) the alpha == beta
    channel, which interferes with the neutral-current diagram and
    requires the unified formula of Section 3.1.2 (eq. 3.4) instead, and
    (b) all semileptonic/hadronic channels (Section 3.2), which typically
    dominate Gamma_N once they are kinematically open. It is used here as
    a leading-order estimate of Gamma_N for the decay-volume survival
    probability, not a full branching-ratio calculation.
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    total = 0.0
    for alpha, m_alpha in LEPTON_MASS.items():
        if U2[alpha] == 0.0:
            continue
        x_l = m_alpha / mN
        for beta, m_beta in LEPTON_MASS.items():
            if beta == alpha:
                continue  # interferes with the NC diagram; see eq. (3.4)
            x_d = m_beta / mN  # charged lepton l_beta^+
            x_u = 0.0          # neutrino nu_beta, effectively massless
            I_val = _cc_phase_space_integral(x_u, x_d, x_l)
            if I_val <= 0.0:
                continue
            total += G_F**2 * mN**5 / (192 * np.pi**3) * U2[alpha] * I_val

    # Majorana HNL: the charge-conjugated channels are not included in the
    # sum above and contribute equally (Bondarenko et al., Section 3.1).
    return 2.0 * total


SIN2_THETA_W = 0.23122  # effective weak mixing angle (PDG)

# "Current" quark masses [GeV] (PDG, MSbar), used only for phase-space
# thresholds in the quark-level CC/NC decay widths below.
QUARK_MASS = {"u": 0.0022, "d": 0.0047, "s": 0.095, "c": 1.27, "b": 4.18}

# (up-type mass, down-type mass, |V_ij|) for each CKM-allowed CC quark pair
# N -> l_alpha^- u_i dbar_j. Top quark omitted (always kinematically closed
# at these HNL masses).
CKM_QUARK_PAIRS = [
    (QUARK_MASS["u"], QUARK_MASS["d"], 0.97435),
    (QUARK_MASS["u"], QUARK_MASS["s"], 0.22500),
    (QUARK_MASS["u"], QUARK_MASS["b"], 0.00369),
    (QUARK_MASS["c"], QUARK_MASS["d"], 0.22486),
    (QUARK_MASS["c"], QUARK_MASS["s"], 0.97349),
    (QUARK_MASS["c"], QUARK_MASS["b"], 0.04182),
]

UP_TYPE_QUARK_MASSES = (QUARK_MASS["u"], QUARK_MASS["c"])
DOWN_TYPE_QUARK_MASSES = (QUARK_MASS["d"], QUARK_MASS["s"], QUARK_MASS["b"])


def _nc_table3_coefficients(fermion_kind: str):
    """C1, C2 coefficients of Table 3 (Bondarenko et al.) for the
    neutral-current-mediated decay width, eq. (3.4)."""
    s2 = SIN2_THETA_W
    if fermion_kind == "up":
        C1 = 0.25 * (1 - 8 / 3 * s2 + 32 / 9 * s2**2)
        C2 = 1 / 3 * s2 * (4 / 3 * s2 - 1)
    elif fermion_kind == "down":
        C1 = 0.25 * (1 - 4 / 3 * s2 + 8 / 9 * s2**2)
        C2 = 1 / 6 * s2 * (2 / 3 * s2 - 1)
    elif fermion_kind == "lepton_diff":
        C1 = 0.25 * (1 - 4 * s2 + 8 * s2**2)
        C2 = 0.5 * s2 * (2 * s2 - 1)
    elif fermion_kind == "lepton_same":
        C1 = 0.25 * (1 + 4 * s2 + 8 * s2**2)
        C2 = 0.5 * s2 * (2 * s2 + 1)
    else:
        raise ValueError(fermion_kind)
    return C1, C2


def _nc_log_function(x: float) -> float:
    """
    L(x) of Bondarenko et al. eq. (3.4), rewritten to avoid catastrophic
    cancellation at small x (e.g. light quarks/leptons): the naive
    numerator 1 - 3x^2 - (1-x^2)*sqrt(1-4x^2) is a difference of two O(1)
    terms that agree through O(x^4), so it is rationalized here into the
    algebraically equivalent 4x^6 / (1 - 3x^2 + (1-x^2)*sqrt(1-4x^2)),
    which stays numerically stable down to arbitrarily small x > 0.
    """
    beta = np.sqrt(1.0 - 4.0 * x**2)
    numerator = 4.0 * x**6 / (1.0 - 3.0 * x**2 + (1.0 - x**2) * beta)
    denominator = x**2 * (1.0 + beta)
    return np.log(numerator / denominator)


def _nc_decay_width(mN: float, m_f: float, N_Z: int, fermion_kind: str, U2_alpha: float) -> float:
    """Neutral-current-mediated width Gamma(N -> nu_alpha f fbar), eq. (3.4)."""
    x = m_f / mN
    if x >= 0.5 or U2_alpha == 0.0:
        return 0.0
    C1, C2 = _nc_table3_coefficients(fermion_kind)
    beta = np.sqrt(1.0 - 4.0 * x**2)
    Lx = _nc_log_function(x)
    term1 = C1 * ((1 - 14 * x**2 - 2 * x**4 - 12 * x**6) * beta + 12 * x**4 * (x**4 - 1) * Lx)
    term2 = 4 * C2 * (x**2 * (2 + 10 * x**2 - 12 * x**4) * beta + 6 * x**4 * (1 - 2 * x**2 + 2 * x**4) * Lx)
    return N_Z * G_F**2 * mN**5 / (192 * np.pi**3) * U2_alpha * (term1 + term2)


def nc_leptonic_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Neutral-current-mediated visible leptonic width, Gamma(N -> nu_alpha
    l_beta^- l_beta^+), summed over all alpha, beta (Bondarenko et al.
    Section 3.1.2, eq. 3.4). The beta == alpha row of Table 3 already
    includes the full charged-current/neutral-current interference for
    that channel (see the main text above eq. 3.4), so it is not
    double-counted against cc_leptonic_decay_width, which explicitly
    excludes beta == alpha.
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    total = 0.0
    for alpha in LEPTON_MASS:
        if U2[alpha] == 0.0:
            continue
        for beta, m_beta in LEPTON_MASS.items():
            kind = "lepton_same" if beta == alpha else "lepton_diff"
            total += _nc_decay_width(mN, m_beta, 1, kind, U2[alpha])
    return 2.0 * total  # Majorana charge-conjugate channels


def invisible_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Neutral-current decay into three neutrinos, Gamma(N -> nu_alpha nu_beta
    nu_beta_bar), summed over alpha, beta (Bondarenko et al. eq. 3.5).
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    total = 0.0
    for alpha in LEPTON_MASS:
        if U2[alpha] == 0.0:
            continue
        for beta in LEPTON_MASS:
            delta = 1.0 if beta == alpha else 0.0
            total += (1 + delta) * G_F**2 * mN**5 / (768 * np.pi**3) * U2[alpha]
    return 2.0 * total


def cc_semileptonic_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Charged-current-mediated semileptonic width, Gamma(N -> l_alpha^-
    u_i dbar_j), summed over all kinematically open (alpha, i, j).
    Bondarenko et al. Section 3.1.1, eq. (3.1), with N_W = N_c|V_ij|^2 = 3|V_ij|^2.
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    total = 0.0
    for alpha, m_alpha in LEPTON_MASS.items():
        if U2[alpha] == 0.0:
            continue
        x_l = m_alpha / mN
        for m_u, m_d, Vij in CKM_QUARK_PAIRS:
            I_val = _cc_phase_space_integral(m_u / mN, m_d / mN, x_l)
            if I_val <= 0.0:
                continue
            N_W = 3 * Vij**2
            total += N_W * G_F**2 * mN**5 / (192 * np.pi**3) * U2[alpha] * I_val
    return 2.0 * total  # Majorana charge-conjugate channels


def nc_quark_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """Neutral-current-mediated quark-pair width, Gamma(N -> nu_alpha q qbar),
    summed over up- and down-type quarks (Bondarenko et al. eq. 3.4)."""
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    total = 0.0
    for alpha in LEPTON_MASS:
        if U2[alpha] == 0.0:
            continue
        for m_f in UP_TYPE_QUARK_MASSES:
            total += _nc_decay_width(mN, m_f, 3, "up", U2[alpha])
        for m_f in DOWN_TYPE_QUARK_MASSES:
            total += _nc_decay_width(mN, m_f, 3, "down", U2[alpha])
    return 2.0 * total


# Charged/neutral pseudoscalar meson masses [GeV] (PDG) and decay constants
# [GeV] (Bondarenko et al. Table 8, values from lattice/experiment [94]).
PSEUDOSCALAR_MESON = {
    # name: (mass, decay_constant)
    "pi+": (0.13957, 0.1302),
    "pi0": (0.13498, 0.1302),   # == f_pi+ by isospin symmetry (App. C.1.1)
    "K+":  (0.49368, 0.1556),
    "eta": (0.54786, 0.0817),
    "etap": (0.95778, 0.0947),  # |f_eta'|; only f_h^2 enters the width
}

CKM_VUD = 0.97435  # |V_ud|, matches CKM_QUARK_PAIRS above
CKM_VUS = 0.22500  # |V_us|, matches CKM_QUARK_PAIRS above


def _pseudoscalar_charged_decay_width(mN: float, m_h: float, f_h: float, Vud: float,
                                       U2_alpha: float, x_l: float) -> float:
    """Gamma(N -> l_alpha^- h_P^+) for a charged pseudoscalar meson h_P^+,
    Bondarenko et al. eq. (3.6)."""
    x_h = m_h / mN
    if x_l + x_h >= 1.0:
        return 0.0  # kinematically closed
    lam = _kallen(1.0, x_h**2, x_l**2)
    bracket = (1 - x_l**2) ** 2 - x_h**2 * (1 + x_l**2)
    if lam <= 0.0 or bracket <= 0.0:
        return 0.0
    return (G_F**2 * f_h**2 * Vud**2 * U2_alpha * mN**3 / (16 * np.pi)
            * bracket * np.sqrt(lam))


def _pseudoscalar_neutral_decay_width(mN: float, m_h: float, f_h: float, U2_alpha: float) -> float:
    """Gamma(N -> nu_alpha h_P^0) for a neutral pseudoscalar meson h_P^0,
    Bondarenko et al. eq. (3.7)."""
    x_h = m_h / mN
    if x_h >= 1.0:
        return 0.0
    return G_F**2 * f_h**2 * mN**3 / (32 * np.pi) * U2_alpha * (1 - x_h**2) ** 2


def exclusive_meson_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    HNL decay width into the lightest exclusive hadronic final states --
    pi^+/-, pi^0, K^+/-, eta, eta' -- from Bondarenko et al. Section 3.2.1,
    eqs. (3.6)-(3.7). This is the physically correct picture near/just
    above threshold, where the quark pair binds into a single meson rather
    than behaving like free partons (contrast the quark-level treatment in
    _quark_hadronic_decay_width below, valid only for M_N >~ 1 GeV).

    Missing relative to a full Section 3.2.1 treatment: the vector mesons
    (rho, a1, D*, ...) and eta_c/D/B channels -- not needed below 1 GeV,
    where this function is actually used (see hadronic_decay_width).
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    m_piplus, f_pi = PSEUDOSCALAR_MESON["pi+"]
    m_pi0, _ = PSEUDOSCALAR_MESON["pi0"]
    m_K, f_K = PSEUDOSCALAR_MESON["K+"]
    m_eta, f_eta = PSEUDOSCALAR_MESON["eta"]
    m_etap, f_etap = PSEUDOSCALAR_MESON["etap"]

    total = 0.0
    for alpha, m_alpha in LEPTON_MASS.items():
        if U2[alpha] == 0.0:
            continue
        x_l = m_alpha / mN
        # charged pseudoscalars: N -> l_alpha^- h_P^+
        total += _pseudoscalar_charged_decay_width(mN, m_piplus, f_pi, CKM_VUD, U2[alpha], x_l)
        total += _pseudoscalar_charged_decay_width(mN, m_K, f_K, CKM_VUS, U2[alpha], x_l)
        # neutral pseudoscalars: N -> nu_alpha h_P^0
        total += _pseudoscalar_neutral_decay_width(mN, m_pi0, f_pi, U2[alpha])
        total += _pseudoscalar_neutral_decay_width(mN, m_eta, f_eta, U2[alpha])
        total += _pseudoscalar_neutral_decay_width(mN, m_etap, f_etap, U2[alpha])
    return 2.0 * total  # Majorana charge-conjugate channels


LAMBDA_QCD_4F = 0.325  # GeV, 4-flavor QCD scale (PDG-ish), used in alpha_s below
HADRONIC_WIDTH_VALIDITY_FLOOR = 1.0  # GeV; see hadronic_decay_width docstring


def alpha_s(Q: float, n_f: int = 4) -> float:
    """One-loop running strong coupling, clipped to avoid blowing up near Lambda_QCD."""
    if Q <= 1.5 * LAMBDA_QCD_4F:
        return 0.5
    beta0 = 11 - 2 * n_f / 3
    return min(4 * np.pi / (beta0 * np.log((Q / LAMBDA_QCD_4F) ** 2)), 0.5)


def delta_qcd(mN: float) -> float:
    """QCD correction factor Delta_QCD to the tree-level quark width, eq. (3.11)."""
    a = alpha_s(mN) / np.pi
    return a + 5.2 * a**2 + 26.4 * a**3


def _quark_hadronic_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """Tree-level CC+NC quark-pair width with the QCD loop correction
    applied (Bondarenko et al. Section 3.2.2, eqs. 3.10-3.11)."""
    tree_level = (cc_semileptonic_decay_width(mN, Ue2, Umu2, Utau2)
                  + nc_quark_decay_width(mN, Ue2, Umu2, Utau2))
    return tree_level * (1.0 + delta_qcd(mN))


def hadronic_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Estimate of the full HNL hadronic decay width, Gamma_had.

    Below HADRONIC_WIDTH_VALIDITY_FLOOR (M_N ~< 1 GeV), quarks are
    confined and the physically correct description is the exclusive
    single-meson channels of Section 3.2.1 (exclusive_meson_decay_width):
    pi, K, eta, eta'.

    At/above the floor, this switches to the quark-level tree-level CC+NC
    width with the QCD loop correction applied (_quark_hadronic_decay_width,
    Section 3.2.2), which tracks the *total* hadronic width including
    multi-meson final states that the exclusive channels above don't
    cover, and is the range where Bondarenko et al. validate this
    quark-level estimate (Figs. 13, 16).

    The two pieces are not summed (that would double-count: the quark-level
    width already includes the pi/K/eta content inclusively), just switched
    at the floor -- so there is a real, expected discontinuity in Gamma_N
    at HADRONIC_WIDTH_VALIDITY_FLOOR, reflecting the matching between two
    different descriptions rather than a smooth interpolation.
    """
    if mN < HADRONIC_WIDTH_VALIDITY_FLOOR:
        return exclusive_meson_decay_width(mN, Ue2, Umu2, Utau2)
    return _quark_hadronic_decay_width(mN, Ue2, Umu2, Utau2)


def total_decay_width(mN: float, Ue2: float, Umu2: float, Utau2: float) -> float:
    """
    Total HNL decay width Gamma_N [GeV], combining every channel
    implemented from Bondarenko et al. (arXiv:1805.08567) Section 3:
    charged- and neutral-current leptonic decays and the invisible
    neutrino channel (Section 3.1), plus the quark-level hadronic estimate
    (Section 3.2.2, valid for M_N >~ 1 GeV -- see hadronic_decay_width).
    """
    return (
        cc_leptonic_decay_width(mN, Ue2, Umu2, Utau2)
        + nc_leptonic_decay_width(mN, Ue2, Umu2, Utau2)
        + invisible_decay_width(mN, Ue2, Umu2, Utau2)
        + hadronic_decay_width(mN, Ue2, Umu2, Utau2)
    )


def list_decay_channels(mN: float, Ue2: float, Umu2: float, Utau2: float) -> list:
    """
    List the decay channels that actually contribute to
    total_decay_width(mN, Ue2, Umu2, Utau2): only channels with a nonzero
    mixing angle for the relevant lepton flavor AND that are kinematically
    open at this mN are included. Mirrors the channel enumeration and
    threshold checks used by the *_decay_width functions above, reusing
    their low-level helpers so the two can't drift out of sync, but without
    recomputing/reporting the numeric widths themselves.
    """
    U2 = {11: Ue2, 13: Umu2, 15: Utau2}
    channels = []

    # Charged-current leptonic (Section 3.1.1, alpha != beta)
    for alpha, m_alpha in LEPTON_MASS.items():
        if U2[alpha] == 0.0:
            continue
        x_l = m_alpha / mN
        for beta, m_beta in LEPTON_MASS.items():
            if beta == alpha:
                continue
            if _cc_phase_space_integral(0.0, m_beta / mN, x_l) > 0.0:
                channels.append(
                    f"N -> {LEPTON_NAME[alpha]}- nu_{LEPTON_NAME[beta]} {LEPTON_NAME[beta]}+ (CC)"
                )

    # Neutral-current leptonic (Section 3.1.2, all beta incl. beta == alpha)
    for alpha in LEPTON_MASS:
        if U2[alpha] == 0.0:
            continue
        for beta, m_beta in LEPTON_MASS.items():
            kind = "lepton_same" if beta == alpha else "lepton_diff"
            if _nc_decay_width(mN, m_beta, 1, kind, U2[alpha]) > 0.0:
                channels.append(
                    f"N -> nu_{LEPTON_NAME[alpha]} {LEPTON_NAME[beta]}- {LEPTON_NAME[beta]}+ (NC)"
                )

    # Invisible (eq. 3.5): always kinematically open (massless neutrinos)
    for alpha in LEPTON_MASS:
        if U2[alpha] == 0.0:
            continue
        for beta in LEPTON_MASS:
            channels.append(
                f"N -> nu_{LEPTON_NAME[alpha]} nu_{LEPTON_NAME[beta]} nu_{LEPTON_NAME[beta]}bar (invisible)"
            )

    # Hadronic: exclusive single-meson channels below the quark-level floor,
    # quark-level CC+NC (with QCD correction) at/above it -- see
    # hadronic_decay_width for why these are switched, not summed.
    if mN < HADRONIC_WIDTH_VALIDITY_FLOOR:
        m_piplus, f_pi = PSEUDOSCALAR_MESON["pi+"]
        m_pi0, _ = PSEUDOSCALAR_MESON["pi0"]
        m_K, f_K = PSEUDOSCALAR_MESON["K+"]
        m_eta, f_eta = PSEUDOSCALAR_MESON["eta"]
        m_etap, f_etap = PSEUDOSCALAR_MESON["etap"]
        for alpha, m_alpha in LEPTON_MASS.items():
            if U2[alpha] == 0.0:
                continue
            x_l = m_alpha / mN
            if _pseudoscalar_charged_decay_width(mN, m_piplus, f_pi, CKM_VUD, U2[alpha], x_l) > 0.0:
                channels.append(f"N -> {LEPTON_NAME[alpha]}- pi+ (exclusive hadronic, CC)")
            if _pseudoscalar_charged_decay_width(mN, m_K, f_K, CKM_VUS, U2[alpha], x_l) > 0.0:
                channels.append(f"N -> {LEPTON_NAME[alpha]}- K+ (exclusive hadronic, CC)")
            if _pseudoscalar_neutral_decay_width(mN, m_pi0, f_pi, U2[alpha]) > 0.0:
                channels.append(f"N -> nu_{LEPTON_NAME[alpha]} pi0 (exclusive hadronic, NC)")
            if _pseudoscalar_neutral_decay_width(mN, m_eta, f_eta, U2[alpha]) > 0.0:
                channels.append(f"N -> nu_{LEPTON_NAME[alpha]} eta (exclusive hadronic, NC)")
            if _pseudoscalar_neutral_decay_width(mN, m_etap, f_etap, U2[alpha]) > 0.0:
                channels.append(f"N -> nu_{LEPTON_NAME[alpha]} eta' (exclusive hadronic, NC)")
    else:
        quark_name = {mass: name for name, mass in QUARK_MASS.items()}
        for alpha, m_alpha in LEPTON_MASS.items():
            if U2[alpha] == 0.0:
                continue
            x_l = m_alpha / mN
            for m_u, m_d, Vij in CKM_QUARK_PAIRS:
                if _cc_phase_space_integral(m_u / mN, m_d / mN, x_l) > 0.0:
                    channels.append(
                        f"N -> {LEPTON_NAME[alpha]}- {quark_name[m_u]} "
                        f"{quark_name[m_d]}bar (hadronic, quark-level CC)"
                    )
        for alpha in LEPTON_MASS:
            if U2[alpha] == 0.0:
                continue
            for m_f in UP_TYPE_QUARK_MASSES:
                if _nc_decay_width(mN, m_f, 3, "up", U2[alpha]) > 0.0:
                    channels.append(
                        f"N -> nu_{LEPTON_NAME[alpha]} {quark_name[m_f]} "
                        f"{quark_name[m_f]}bar (hadronic, quark-level NC)"
                    )
            for m_f in DOWN_TYPE_QUARK_MASSES:
                if _nc_decay_width(mN, m_f, 3, "down", U2[alpha]) > 0.0:
                    channels.append(
                        f"N -> nu_{LEPTON_NAME[alpha]} {quark_name[m_f]} "
                        f"{quark_name[m_f]}bar (hadronic, quark-level NC)"
                    )

    return channels


def load_events(root_path: str, npot_path: str):
    """
    Load the HNL ntuple and the number of simulated protons-on-target.
    """
    tree = uproot.open(root_path)["HNL"]
    data = tree.arrays(library="np")

    with open(npot_path) as f:
        n_pot = float(f.read().strip())

    return data, n_pot


def mixing_weight(lepton_pdg: np.ndarray, Ue2: float, Umu2: float, Utau2: float) -> np.ndarray:
    """
    Map each row's accompanying-lepton PDG code to the corresponding
    |U_alpha|^2 for the requested mixing pattern.
    """
    U2 = np.select(
        [np.abs(lepton_pdg) == 11, np.abs(lepton_pdg) == 13, np.abs(lepton_pdg) == 15],
        [Ue2, Umu2, Utau2],
        default=0.0,
    )
    return U2


def add_series(ax, kind: str, x, y, label: str, **kwargs) -> None:
    """
    Draw one series only if it has at least one positive value. A series
    that's all zero (e.g. pi/K above their kinematic threshold mass) is
    invisible on a log-scale axis anyway, so skipping it also keeps its
    swatch out of the legend instead of showing an empty entry.
    """
    if not np.any(y > 0):
        return
    if kind == "plot":
        ax.plot(x, y, label=label, **kwargs)
    elif kind == "stairs":
        ax.stairs(y, x, label=label, **kwargs)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def legend_if_any(ax, **kwargs) -> None:
    """Only call ax.legend() if at least one series was actually drawn with a label."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def flux_vs_mass_table(data: dict, n_pot: float, Ue2: float, Umu2: float, Utau2: float) -> pd.DataFrame:
    """
    Flux [N/POT] vs. HNL benchmark mass, one column per parent meson PDG
    plus 'light' (pi/K), 'heavy' (D/Ds/B/Bc), and 'total'.
    """
    weight = data["weightPerU2"] * mixing_weight(data["leptonPDG"], Ue2, Umu2, Utau2) / n_pot
    df = pd.DataFrame({"mN": data["mN"], "parentPDG": np.abs(data["parentPDG"]), "weight": weight})

    table = df.groupby(["mN", "parentPDG"])["weight"].sum().unstack(fill_value=0.0)
    table = table.reindex(columns=list(PARENT_LABEL.keys()), fill_value=0.0).sort_index()

    table["light"] = table[list(LIGHT_PARENTS)].sum(axis=1)
    table["heavy"] = table[list(HEAVY_PARENTS)].sum(axis=1)
    table["total"] = table["light"] + table["heavy"]

    return table


def plot_flux_vs_mass(data: dict, n_pot: float, Ue2: float, Umu2: float, Utau2: float,
                       outfile: str, breakdown: bool = False) -> None:
    """
    Total HNL production flux per POT, summed over all channels, as a
    function of the HNL benchmark mass.
    """
    table = flux_vs_mass_table(data, n_pot, Ue2, Umu2, Utau2)
    mass_points = table.index.to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)

    add_series(ax, "plot", mass_points, table["total"].to_numpy(), "Total", color="k", lw=5)

    if breakdown:
        for pdg, label in PARENT_LABEL.items():
            add_series(ax, "plot", mass_points, table[pdg].to_numpy(), label,
                       color=PARENT_COLORS[pdg], lw=4, ls="--")
    else:
        add_series(ax, "plot", mass_points, table["light"].to_numpy(), r"$\pi^\pm, K^\pm$",
                   color="#00B945", lw=5, ls="--")
        add_series(ax, "plot", mass_points, table["heavy"].to_numpy(), r"$D^\pm, D_s^\pm, B^\pm, B_c^\pm$",
                   color="#FF2C00", lw=5, ls="--")

    ax.set_xlabel(r"HNL mass $M_N$ [GeV]")
    ax.set_ylabel(r"HNL flux $\Phi_N$ [$N$ / POT]")
    ax.set_title(
        rf"{{\bf At proton target}} ($U_e^2={Ue2:.2g},\ U_\mu^2={Umu2:.2g},\ U_\tau^2={Utau2:.2g}$)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    set_log_ticks(ax)
    legend_if_any(ax, loc="upper right")
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)

    fig.savefig(outfile, dpi=100)


def energy_spectrum_table(data: dict, n_pot: float, target_mass: float,
                           Ue2: float, Umu2: float, Utau2: float):
    """
    Differential flux dPhi/dE_N [N/GeV/POT] at the mass grid point closest
    to target_mass, one array per parent meson PDG plus 'light' and
    'heavy', all sharing the same (log-spaced) bin edges.
    Also calculates differential flux at decay volume.
    """
    mass_points = np.unique(data["mN"])
    closest_mass = mass_points[np.argmin(np.abs(mass_points - target_mass))]

    in_bin = data["mN"] == closest_mass
    weight = data["weightPerU2"][in_bin] * mixing_weight(data["leptonPDG"][in_bin], Ue2, Umu2, Utau2) / n_pot
    energy = data["E_N"][in_bin]
    parent = np.abs(data["parentPDG"][in_bin])

    e_max = max(energy.max(), 1.0)
    bins = np.geomspace(max(energy[energy > 0].min(), 1e-3), e_max, 40)
    bin_widths = np.diff(bins)

    def diff_flux(mask: np.ndarray, weights: np.ndarray = weight) -> np.ndarray:
        # np.histogram gives the flux *integrated within each bin* (N/POT);
        # dividing by the bin width turns that into a true differential
        # quantity dPhi/dE_N (N/GeV/POT) -- necessary here because the bins
        # are log-spaced (width grows with energy), so skipping this step
        # would distort the spectrum shape, not just its overall scale.
        counts, _ = np.histogram(energy[mask], bins=bins, weights=weights[mask])
        return counts / bin_widths

    spectra = {pdg: diff_flux(parent == pdg) for pdg in PARENT_LABEL}
    spectra["light"] = diff_flux(np.isin(parent, LIGHT_PARENTS))
    spectra["heavy"] = diff_flux(np.isin(parent, HEAVY_PARENTS))

    # decay volume flux implementation
    px_N = data["px_N"][in_bin] # GeV
    py_N = data["py_N"][in_bin] # GeV
    pz_N = data["pz_N"][in_bin] # GeV

    vx = data["vx"][in_bin] # mm 
    vy = data["vy"][in_bin] # mm
    vz = data["vz"][in_bin] # mm

    # HNL vertex at decay volume. Undefined/meaningless for pz_N <= 0 --
    # excluded below via forward_going before it can be used.
    vx_DV = vx + (Z_DECAY_VOLUME - vz) * px_N / pz_N  # mm
    vy_DV = vy + (Z_DECAY_VOLUME - vz) * py_N / pz_N  # mm

    # apply geometrical cut ('and'/'or' don't broadcast over numpy arrays;
    # need the elementwise '&' operator here). forward_going is required
    # first: for pz_N <= 0 the extrapolated position above is not
    # physically meaningful (a backward/transverse-going HNL can never
    # reach a downstream decay volume), and the transverse coordinates can
    # spuriously fall inside the acceptance window by coincidence.
    forward_going = pz_N > 0
    x_acceptance = X_DECAY_VOLUME > np.abs(vx_DV)
    y_acceptance = Y_DECAY_VOLUME > np.abs(vy_DV)
    geom_acceptance = forward_going & x_acceptance & y_acceptance

    # calculate survival probability (only defined/needed within the
    # geometrical acceptance; computed over the full array first so it can
    # be combined later with other full-length masks like 'parent')
    mm_to_invGeV = 5.07e12 # 1 mm = 5.07e12 GeV^-1
    t = (Z_DECAY_VOLUME - vz) * energy / pz_N * mm_to_invGeV # GeV^-1
    lorentz_factor = energy / closest_mass
    decay_width = total_decay_width(closest_mass, Ue2, Umu2, Utau2)  # GeV

    survival_probability = np.zeros_like(energy)
    survival_probability[geom_acceptance] = np.exp(
        -t[geom_acceptance] * decay_width / lorentz_factor[geom_acceptance]
    )

    # Scale each event's existing weight by its exact survival probability,
    # rather than an accept-reject coin flip: the latter would add
    # unnecessary sampling variance on top of an already-known probability,
    # and (since np.random isn't seeded here) would make the resulting plot
    # non-reproducible from run to run. Entries outside geom_acceptance
    # already carry survival_probability == 0, so no extra masking is
    # needed here.
    weight_DV = weight * survival_probability

    spectra_DV = {pdg: diff_flux(parent == pdg, weight_DV) for pdg in PARENT_LABEL}
    spectra_DV["light"] = diff_flux(np.isin(parent, LIGHT_PARENTS), weight_DV)
    spectra_DV["heavy"] = diff_flux(np.isin(parent, HEAVY_PARENTS), weight_DV)

    return closest_mass, bins, spectra, spectra_DV


def plot_energy_spectrum(data: dict, n_pot: float, target_mass: float,
                          Ue2: float, Umu2: float, Utau2: float, outfile: str, outfile_DV: str,
                          breakdown: bool = False) -> None:
    """
    Differential HNL flux vs. lab energy at the mass grid point closest to
    target_mass, split into light-meson (pi/K) and heavy-flavor (D/Ds/B/Bc)
    contributions (or into individual meson species if breakdown=True).
    """
    closest_mass, bins, spectra, spectra_DV = energy_spectrum_table(data, n_pot, target_mass, Ue2, Umu2, Utau2)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)

    if breakdown:
        for pdg, label in PARENT_LABEL.items():
            add_series(ax, "stairs", bins, spectra[pdg], label, color=PARENT_COLORS[pdg], lw=4)
    else:
        add_series(ax, "stairs", bins, spectra["light"], r"$\pi^\pm, K^\pm$", color="#00B945", lw=5)
        add_series(ax, "stairs", bins, spectra["heavy"], r"$D^\pm, D_s^\pm, B^\pm, B_c^\pm$", color="#FF2C00", lw=5)

    ax.set_xlabel(r"HNL lab energy $E_N$ [GeV]")
    ax.set_ylabel(r"$\mathrm{d}\Phi_N / \mathrm{d}E_N$ [$N$ / GeV / POT]")
    ax.set_title(
        rf"{{\bf At proton target}} ($M_N={closest_mass:.3g}$ GeV, "
        rf"$U_e^2={Ue2:.2g},\ U_\mu^2={Umu2:.2g},\ U_\tau^2={Utau2:.2g}$)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    set_log_ticks(ax)
    legend_if_any(ax, loc="upper right")
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)

    fig.savefig(outfile, dpi=200)

    # flux at decay volume
    fig_DV, ax_DV = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)

    if breakdown:
        for pdg, label in PARENT_LABEL.items():
            add_series(ax_DV, "stairs", bins, spectra_DV[pdg], label, color=PARENT_COLORS[pdg], lw=4)
    else:
        add_series(ax_DV, "stairs", bins, spectra_DV["light"], r"$\pi^\pm, K^\pm$", color="#00B945", lw=5)
        add_series(ax_DV, "stairs", bins, spectra_DV["heavy"], r"$D^\pm, D_s^\pm, B^\pm, B_c^\pm$", color="#FF2C00", lw=5)

    ax_DV.set_xlabel(r"HNL lab energy $E_N$ [GeV]")
    ax_DV.set_ylabel(r"$\mathrm{d}\Phi_N / \mathrm{d}E_N$ [$N$ / GeV / POT]")
    ax_DV.set_title(
        rf"{{\bf At decay volume}} ($M_N={closest_mass:.3g}$ GeV, "
        rf"$U_e^2={Ue2:.2g},\ U_\mu^2={Umu2:.2g},\ U_\tau^2={Utau2:.2g}$)"
    )
    ax_DV.set_xscale("log")
    ax_DV.set_yscale("log")
    set_log_ticks(ax_DV)
    legend_if_any(ax_DV, loc="upper right")
    ax_DV.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)
    ax_DV.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)

    fig_DV.savefig(outfile_DV, dpi=200)

    return closest_mass, bins, spectra, spectra_DV


def write_summary_file(outfile: str, target_mass: float, closest_mass: float,
                        Ue2: float, Umu2: float, Utau2: float,
                        n_pot: float, pot: float,
                        bins: np.ndarray, spectra: dict, spectra_DV: dict) -> None:
    """
    Write a text summary of the run parameters (HNL mass, mixing angles,
    decay volume geometry, POT) and the resulting HNL yields, obtained by
    integrating the differential energy spectra dPhi_N/dE_N (both at the
    target and at the decay volume) over dE_N.
    """
    bin_widths = np.diff(bins)
    n_per_pot_target = float(np.sum((spectra["light"] + spectra["heavy"]) * bin_widths))
    n_per_pot_DV = float(np.sum((spectra_DV["light"] + spectra_DV["heavy"]) * bin_widths))
    decay_width = total_decay_width(closest_mass, Ue2, Umu2, Utau2)

    with open(outfile, "w") as f:
        f.write("HNL flux summary\n")
        f.write("================\n\n")

        f.write("HNL parameters\n")
        f.write("--------------\n")
        f.write(f"Requested mass [GeV]:            {target_mass:.6g}\n")
        f.write(f"Nearest simulated benchmark [GeV]: {closest_mass:.6g}\n")
        f.write(f"Mixing: Ue2={Ue2:.6g}, Umu2={Umu2:.6g}, Utau2={Utau2:.6g}\n")
        f.write(f"Total decay width Gamma_N [GeV]: {decay_width:.6g}\n\n")

        f.write("Decay channels included (kinematically open at this mass and mixing)\n")
        f.write("----------------------------------------------------------------------\n")
        channels = list_decay_channels(closest_mass, Ue2, Umu2, Utau2)
        if channels:
            for channel in channels:
                f.write(f"  {channel}\n")
        else:
            f.write("  (none -- Gamma_N == 0 at this mass/mixing)\n")
        f.write("\n")

        f.write("Decay volume geometry\n")
        f.write("----------------------\n")
        f.write(f"Distance from target [m]: {Z_DECAY_VOLUME / 1e3:.6g}\n")
        f.write(f"Half-width (x) [m]:       {X_DECAY_VOLUME / 1e3:.6g}\n")
        f.write(f"Half-height (y) [m]:      {Y_DECAY_VOLUME / 1e3:.6g}\n")
        f.write(f"Cross-sectional area [m^2]: {2 * X_DECAY_VOLUME / 1e3 * 2 * Y_DECAY_VOLUME / 1e3:.6g}\n\n")

        f.write("Protons on target\n")
        f.write("-----------------\n")
        f.write(f"Simulated (MC normalization) POT: {n_pot:.6g}\n")
        f.write(f"Requested (scaling) POT:          {pot:.6g}\n\n")

        f.write("HNL yields (light + heavy parents, integrated over dPhi_N/dE_N)\n")
        f.write("----------------------------------------------------------------\n")
        f.write(f"Flux at target [N/POT]:        {n_per_pot_target:.6g}\n")
        f.write(f"Flux at decay volume [N/POT]:  {n_per_pot_DV:.6g}\n")
        f.write(f"Total HNLs produced at target for {pot:.6g} POT: {n_per_pot_target * pot:.6g}\n")
        f.write(f"Total HNLs detected at decay volume for {pot:.6g} POT: {n_per_pot_DV * pot:.6g}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="../geant4/build/HNL_target_flux.root",
                         help="Path to the Geant4 output ROOT file.")
    parser.add_argument("--npot", default="../geant4/build/n_pot.txt",
                         help="Path to the companion protons-on-target count.")
    parser.add_argument("--mass", type=float, default=1.0,
                         help="HNL mass [GeV] for the energy-spectrum plot "
                              "(snaps to the nearest simulated benchmark mass).")
    parser.add_argument("--Ue2", type=float, default=0.0, help="|U_e|^2 mixing.")
    parser.add_argument("--Umu2", type=float, default=1.0, help="|U_mu|^2 mixing.")
    parser.add_argument("--Utau2", type=float, default=0.0, help="|U_tau|^2 mixing.")
    parser.add_argument("--pot", type=float, default=None,
                         help="Protons-on-target to scale the total HNL yield by "
                              "(defaults to the simulated POT count from --npot, "
                              "i.e. no additional scaling).")
    parser.add_argument("--outdir", default="../plots", help="Output directory for plots.")
    parser.add_argument("--breakdown", action="store_true",
                         help="Plot each parent meson species individually instead of "
                              "the aggregated light (pi/K) / heavy (D/Ds/B/Bc) curves.")
    parser.add_argument("--tag", type=str, default=None, help="Tag to append at end of figure names")
    args = parser.parse_args()

    data, n_pot = load_events(args.root, args.npot)
    pot = args.pot if args.pot is not None else n_pot

    if args.tag is not None:
        plot_flux_vs_mass(data, n_pot, args.Ue2, args.Umu2, args.Utau2,
                          f"{args.outdir}/HNL_target_flux_vs_mass_{args.tag}.pdf", breakdown=args.breakdown)
        closest_mass, bins, spectra, spectra_DV = plot_energy_spectrum(
            data, n_pot, args.mass, args.Ue2, args.Umu2, args.Utau2,
            f"{args.outdir}/HNL_target_energy_spectrum_{args.tag}.pdf",
            f"{args.outdir}/HNL_decay_volume_energy_spectrum_{args.tag}.pdf",
            breakdown=args.breakdown)
        write_summary_file(f"{args.outdir}/HNL_summary_{args.tag}.txt",
                            args.mass, closest_mass, args.Ue2, args.Umu2, args.Utau2,
                            n_pot, pot, bins, spectra, spectra_DV)
    else:
        plot_flux_vs_mass(data, n_pot, args.Ue2, args.Umu2, args.Utau2,
                          f"{args.outdir}/HNL_target_flux_vs_mass.pdf", breakdown=args.breakdown)
        closest_mass, bins, spectra, spectra_DV = plot_energy_spectrum(
            data, n_pot, args.mass, args.Ue2, args.Umu2, args.Utau2,
            f"{args.outdir}/HNL_target_energy_spectrum.pdf",
            f"{args.outdir}/HNL_decay_volume_energy_spectrum.pdf",
            breakdown=args.breakdown)
        write_summary_file(f"{args.outdir}/HNL_summary.txt",
                            args.mass, closest_mass, args.Ue2, args.Umu2, args.Utau2,
                            n_pot, pot, bins, spectra, spectra_DV)


if __name__ == "__main__":
    main()
