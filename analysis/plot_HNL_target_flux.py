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
import os
from zlib import Z_DEFAULT_COMPRESSION, Z_DEFAULT_STRATEGY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
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
# Z_DECAY_VOLUME = 33.5e3  # distance from the target (z=0) to the decay volume
# X_DECAY_VOLUME = 0.5e3   # half width
# Y_DECAY_VOLUME = 1.35e3  # half height

# Xiaolin's parameters
Z_DECAY_VOLUME = 35e3 # located 35 m downstream from target
X_DECAY_VOLUME = 2e3 # width of 4 m
Y_DECAY_VOLUME = 3e3 # height of 6 m
DECAY_VOLUME_LENGTH = 50e3  # length of the decay volume along z (SHiP: 50 m)

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


# Table 1 of arXiv:1811.00930: quark-pair production fraction per inelastic
# pN collision and the cascade-production enhancement factor, by flavor.
# Used only to strip the production-rate normalization out of our per-POT
# flux in plot_production_fraction_vs_mass, below -- NOT used anywhere else
# in this file, and not a claim that our simulation's own production rate
# matches these numbers (see that function's docstring).
PAPER_X_QQBAR = {"charm": 1.7e-3, "beauty": 1.6e-7}
PAPER_F_CASCADE = {"charm": 2.3, "beauty": 1.7}


def plot_production_fraction_vs_mass(data: dict, n_pot: float, outfile: str,
                                      mass_max_charm: float = 2.0, mass_max_beauty: float = 6.0) -> None:
    """
    Reproduction of Fig. 2 of the SHiP sensitivity paper (Collaboration,
    arXiv:1811.00930): f(h) BR(h -> N+X) vs HNL mass for each parent meson
    species, with pure electron mixing (Ue2=1, Umu2=Utau2=0) as in that
    figure, split into a charm-meson panel (left) and beauty-meson panel
    (right).

    Normalization: table[pdg] from flux_vs_mass_table is a *per-POT flux*
    -- it includes the full quark-pair production rate (X_qqbar) and
    cascade enhancement, in addition to fragmentation and the HNL branching
    ratio. Fig. 2 explicitly does NOT include the production-rate piece:
    per the paper's eq. (2.2)-(2.3) and the Fig. 2 caption ("production
    fraction of the meson decaying into HNL" = f(q->h), Table 2 alone),
    N_prod = N_q * f(q->h) * BR(h->N+X), with N_q = 2*X_qqbar*f_cascade*POT
    applied *separately* and outside what Fig. 2 plots. So to land in the
    same convention, our per-POT flux is divided here by 2*X_qqbar*f_cascade
    (PAPER_X_QQBAR, PAPER_F_CASCADE above, from the paper's own Table 1) to
    strip the production-rate factor back out.

    Caveat -- this uses the *paper's* assumed production rate, not a
    verified property of our own simulation: dividing by our own simulated
    production rate instead (recovered from a single data point via the
    independently-coded HNLProduction.cc branching-ratio formula) shows our
    effective per-POT Ds+ yield is ~260x lower than 2*X_qqbar*f_cascade*0.088
    (Table 1 x Table 2) would predict -- i.e. there is a real, unresolved
    gap in our simulation's charm/beauty production-rate normalization
    (not in the HNL branching-ratio formula itself, which checks out
    numerically against this same Table 2 fragmentation fraction). This
    plot therefore still will not land on Fig. 2's absolute scale; treat it
    as "our shape, best-effort rescaled by the paper's assumed production
    rate," not a validated match.

    Our Geant4/Pythia8 simulation also only tracks the charged mesons D+,
    Ds+, B+, Bc+ (not the neutral D0, B0, or the vector D*), so this
    reproduces only the subset of curves in Fig. 2 that our simulation
    actually produces -- it is not a full reproduction of that figure.
    """
    table = flux_vs_mass_table(data, n_pot, Ue2=1.0, Umu2=0.0, Utau2=0.0)
    mass_points = table.index.to_numpy()
    mass_max_beauty = min(mass_max_beauty, mass_points.max())

    rescale = {
        411: 1.0 / (2 * PAPER_X_QQBAR["charm"] * PAPER_F_CASCADE["charm"]),
        431: 1.0 / (2 * PAPER_X_QQBAR["charm"] * PAPER_F_CASCADE["charm"]),
        521: 1.0 / (2 * PAPER_X_QQBAR["beauty"] * PAPER_F_CASCADE["beauty"]),
        541: 1.0 / (2 * PAPER_X_QQBAR["beauty"] * PAPER_F_CASCADE["beauty"]),
    }

    fig, (ax_charm, ax_beauty) = plt.subplots(1, 2, figsize=(26, 12), tight_layout=True)

    for pdg in (411, 431):
        add_series(ax_charm, "plot", mass_points, table[pdg].to_numpy() * rescale[pdg], PARENT_LABEL[pdg],
                   color=PARENT_COLORS[pdg], lw=4)
    for pdg in (521, 541):
        add_series(ax_beauty, "plot", mass_points, table[pdg].to_numpy() * rescale[pdg], PARENT_LABEL[pdg],
                   color=PARENT_COLORS[pdg], lw=4)

    for ax, mass_max, title in ((ax_charm, mass_max_charm, "Charm"), (ax_beauty, mass_max_beauty, "Beauty")):
        ax.set_xlim(0.0, mass_max)
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\rm HNL}$ [GeV]")
        ax.set_ylabel(r"$f(h)\,\mathrm{BR}(h \to X+N)$")
        ax.set_title(rf"{{\bf {title}}} ($U_e^2=1,\ U_\mu^2=0,\ U_\tau^2=0$) -- rescaled to paper's convention")
        legend_if_any(ax, loc="upper right")
        ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)

    fig.savefig(outfile, dpi=200)


def _propagate_to_decay_volume(data: dict, in_bin: np.ndarray):
    """
    Extrapolate each selected HNL's production vertex to the decay-volume
    entrance plane (z = Z_DECAY_VOLUME) assuming straight-line propagation,
    returning (vx_DV, vy_DV, forward_going) [mm, mm, bool]. vx_DV/vy_DV are
    only physically meaningful where forward_going (pz_N > 0) is True -- a
    backward/transverse-going HNL can never reach a downstream decay
    volume, so callers must mask on forward_going before using them.
    """
    px_N = data["px_N"][in_bin]  # GeV
    py_N = data["py_N"][in_bin]  # GeV
    pz_N = data["pz_N"][in_bin]  # GeV
    vx = data["vx"][in_bin]  # mm
    vy = data["vy"][in_bin]  # mm
    vz = data["vz"][in_bin]  # mm

    forward_going = pz_N > 0
    vx_DV = vx + (Z_DECAY_VOLUME - vz) * px_N / pz_N  # mm
    vy_DV = vy + (Z_DECAY_VOLUME - vz) * py_N / pz_N  # mm
    return vx_DV, vy_DV, forward_going


def expected_n_events(data: dict, n_pot: float, target_mass: float,
                       Ue2: float, Umu2: float, Utau2: float, pot: float,
                       det_efficiency: float = 1.0):
    """
    Expected number of detected HNL events, following arXiv:1811.00930
    eqs. (2.1), (2.4), (2.5):

        N_events = N_prod * P_decay * BR(N -> visible) * eps_det

    N_prod ("the number of produced HNLs that fly in the direction of the
    fiducial volume") is taken directly from our own simulated, geometry-
    accepted flux -- *not* re-derived via the production decomposition of
    eq. (2.2)-(2.3) (N_q * f(q->h) * BR), since we already have the
    simulated production rate itself and don't need to reconstruct it from
    separate quark-production/fragmentation/branching-ratio factors.

    P_decay (eq. 2.5) is the probability of decaying between the decay
    volume's entrance (Z_DECAY_VOLUME) and exit (Z_DECAY_VOLUME +
    DECAY_VOLUME_LENGTH) planes, computed per simulated HNL via the same
    exact exponent derivation used in energy_spectrum_table's survival
    probability (exponent = Gamma_N * M_N * distance / pz), just evaluated
    at both boundaries and differenced, rather than only at the entrance.

    BR(N -> visible) = 1 - Gamma_invisible/Gamma_N, using the existing
    invisible_decay_width/total_decay_width functions.

    eps_det (detection efficiency: track reconstruction + selection) is not
    modeled here and is fixed at 1.0 by default -- a genuine simplification
    relative to the paper, which derives it from a dedicated FairShip
    reconstruction study.

    Returns (closest_mass, n_prod, n_events).
    """
    mass_points = np.unique(data["mN"])
    closest_mass = mass_points[np.argmin(np.abs(mass_points - target_mass))]
    in_bin = data["mN"] == closest_mass

    weight = data["weightPerU2"][in_bin] * mixing_weight(data["leptonPDG"][in_bin], Ue2, Umu2, Utau2) / n_pot
    vz = data["vz"][in_bin]  # mm
    pz_N = data["pz_N"][in_bin]  # GeV

    vx_DV, vy_DV, forward_going = _propagate_to_decay_volume(data, in_bin)
    x_acceptance = X_DECAY_VOLUME > np.abs(vx_DV)
    y_acceptance = Y_DECAY_VOLUME > np.abs(vy_DV)
    geom_acceptance = forward_going & x_acceptance & y_acceptance

    decay_width = total_decay_width(closest_mass, Ue2, Umu2, Utau2)  # GeV
    mm_to_invGeV = 5.07e12  # 1 mm = 5.07e12 GeV^-1

    p_decay = np.zeros_like(weight)
    exponent_near = (decay_width * closest_mass * (Z_DECAY_VOLUME - vz[geom_acceptance])
                      / pz_N[geom_acceptance] * mm_to_invGeV)
    exponent_far = (decay_width * closest_mass
                     * (Z_DECAY_VOLUME + DECAY_VOLUME_LENGTH - vz[geom_acceptance])
                     / pz_N[geom_acceptance] * mm_to_invGeV)
    p_decay[geom_acceptance] = np.exp(-exponent_near) - np.exp(-exponent_far)

    n_prod_per_pot = np.sum(weight[geom_acceptance])
    n_decaying_per_pot = np.sum(weight * p_decay)  # p_decay already 0 outside geom_acceptance

    br_visible = 1.0 - invisible_decay_width(closest_mass, Ue2, Umu2, Utau2) / decay_width

    n_prod = n_prod_per_pot * pot
    n_events = n_decaying_per_pot * br_visible * det_efficiency * pot

    return closest_mass, n_prod, n_events


FLAVOR_PDG = {"e": 11, "mu": 13, "tau": 15}
FLAVOR_ONEHOT = {"e": (1.0, 0.0, 0.0), "mu": (0.0, 1.0, 0.0), "tau": (0.0, 0.0, 1.0)}


def _n_events_vs_u2_curve(data: dict, n_pot: float, mN: float, flavor: str,
                           pot: float, det_efficiency: float, u2_grid: np.ndarray) -> np.ndarray:
    """
    N_events(mN, U_alpha^2) over a whole grid of trial U_alpha^2 values, for
    pure single-flavor mixing, evaluated efficiently by exploiting that (for
    a single nonzero mixing angle) every term in Gamma_N is linear in
    U_alpha^2: production weight, Gamma_N, and therefore the P_decay
    exponents all scale with u2 given a one-time Gamma_N_per_U2 -- so the
    per-mass event selection and the Gamma_N/BR_visible calculation are each
    done once here, not once per trial u2 as a naive loop over
    expected_n_events would do.
    """
    lepton_pdg = FLAVOR_PDG[flavor]
    onehot = FLAVOR_ONEHOT[flavor]

    in_bin = (data["mN"] == mN) & (np.abs(data["leptonPDG"]) == lepton_pdg)
    if not np.any(in_bin):
        return np.zeros_like(u2_grid)

    weight_per_u2 = data["weightPerU2"][in_bin] / n_pot  # BR/U2 per POT, U2 not yet applied
    vz = data["vz"][in_bin]
    pz_N = data["pz_N"][in_bin]

    vx_DV, vy_DV, forward_going = _propagate_to_decay_volume(data, in_bin)
    x_acceptance = X_DECAY_VOLUME > np.abs(vx_DV)
    y_acceptance = Y_DECAY_VOLUME > np.abs(vy_DV)
    geom_acceptance = forward_going & x_acceptance & y_acceptance
    if not np.any(geom_acceptance):
        return np.zeros_like(u2_grid)

    gamma_per_u2 = total_decay_width(mN, *onehot)  # Gamma_N at U_alpha^2 = 1
    br_visible = 1.0 - invisible_decay_width(mN, *onehot) / gamma_per_u2  # U2-independent ratio

    mm_to_invGeV = 5.07e12
    dist_near = (Z_DECAY_VOLUME - vz[geom_acceptance])
    dist_far = (Z_DECAY_VOLUME + DECAY_VOLUME_LENGTH - vz[geom_acceptance])
    pz_acc = pz_N[geom_acceptance]
    w_acc = weight_per_u2[geom_acceptance]

    n_events = np.empty_like(u2_grid)
    for i, u2 in enumerate(u2_grid):
        gamma_N = u2 * gamma_per_u2
        exponent_near = gamma_N * mN * dist_near / pz_acc * mm_to_invGeV
        exponent_far = gamma_N * mN * dist_far / pz_acc * mm_to_invGeV
        p_decay = np.exp(-exponent_near) - np.exp(-exponent_far)
        n_decaying_per_pot = np.sum(w_acc * u2 * p_decay)
        n_events[i] = n_decaying_per_pot * br_visible * det_efficiency * pot

    return n_events


def _find_sensitivity_boundaries(u2_grid: np.ndarray, n_events_grid: np.ndarray,
                                  target_n_events: float):
    """
    Lower and upper U_alpha^2 roots of N_events(U_alpha^2) == target_n_events
    on a log-spaced u2_grid, found by log-log linear interpolation between
    the bracketing grid points of the first rising crossing (lower boundary)
    and the last falling crossing (upper boundary). Returns (None, None) if
    N_events never reaches target_n_events anywhere on the grid (i.e. this
    mass is outside the sensitivity region for any mixing angle).
    """
    f = n_events_grid - target_n_events
    if not np.any(f > 0):
        return None, None

    log_u2 = np.log10(u2_grid)

    def interp_crossing(i0, i1):
        # f changes sign between grid indices i0 and i1; interpolate in
        # log10(u2) vs log10(n_events) space, since n_events spans many
        # orders of magnitude across the grid.
        y0, y1 = np.log10(n_events_grid[i0]), np.log10(n_events_grid[i1])
        x0, x1 = log_u2[i0], log_u2[i1]
        target = np.log10(target_n_events)
        x_cross = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        return 10 ** x_cross

    sign = np.sign(f)
    crossing_idx = np.where(np.diff(sign) != 0)[0]
    rising = [i for i in crossing_idx if f[i] < 0 < f[i + 1] or (f[i] <= 0 and f[i + 1] > 0)]
    falling = [i for i in crossing_idx if f[i] > 0 > f[i + 1] or (f[i] >= 0 and f[i + 1] < 0)]

    u2_lower = interp_crossing(rising[0], rising[0] + 1) if rising else None
    u2_upper = interp_crossing(falling[-1], falling[-1] + 1) if falling else None
    return u2_lower, u2_upper


def sensitivity_curve(data: dict, n_pot: float, pot: float, flavor: str,
                       det_efficiency: float = 1.0, target_n_events: float = 2.3,
                       u2_bounds=(1e-12, 1.0), n_scan: int = 300):
    """
    SHiP 90% CL sensitivity boundary (arXiv:1811.00930 Fig. 3 construction)
    for pure single-flavor mixing: for each simulated HNL mass, the lower
    and upper U_alpha^2 roots of N_events(mN, U_alpha^2) = target_n_events
    (2.3, the paper's stated 90% CL threshold for ~0.1 expected background
    events). Returns (masses, u2_lower, u2_upper) arrays, NaN where no
    sensitivity exists at that mass (peak N_events below target_n_events
    for any mixing angle in u2_bounds).
    """
    mass_points = np.unique(data["mN"])
    u2_grid = np.geomspace(u2_bounds[0], u2_bounds[1], n_scan)

    u2_lower = np.full(mass_points.shape, np.nan)
    u2_upper = np.full(mass_points.shape, np.nan)

    for i, mN in enumerate(mass_points):
        n_events_grid = _n_events_vs_u2_curve(data, n_pot, mN, flavor, pot, det_efficiency, u2_grid)
        lo, hi = _find_sensitivity_boundaries(u2_grid, n_events_grid, target_n_events)
        if lo is not None:
            u2_lower[i] = lo
        if hi is not None:
            u2_upper[i] = hi

    return mass_points, u2_lower, u2_upper


FIGURE3_CSV_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE3_CSV_VARIANTS = {"with_bc": "withBcproduction", "without_bc": "withoutBcproduction"}


def _split_cigar_branches(mass: np.ndarray, u2: np.ndarray):
    """
    Each SHiP_HNL_<flavor>_mixing_<variant>.csv digitizes a *closed* cigar
    boundary as two separately-monotonic-in-mass curves (lower and upper)
    merged together and sorted by mass into one (mass, U2) sequence -- so
    consecutive rows do not alternate branches in any fixed pattern (confirmed
    by inspection: splitting by even/odd row or by a fixed mass/value gap
    both failed on real examples where two consecutive rows belong to the
    same branch). This disentangles the merge by greedily continuing
    whichever branch (lower/upper) is closer in log10(U2) to the new point,
    processing points in mass order; the upper branch is only "started"
    once a point appears more than a decade above the lower branch's current
    value (matching the real feature that the upper boundary is absent
    entirely -- off the top of the original figure -- below some mass).
    """
    order = np.argsort(mass, kind="stable")
    mass, u2 = mass[order], u2[order]
    log_u2 = np.log10(u2)

    lower_m, lower_u2 = [mass[0]], [u2[0]]
    upper_m, upper_u2 = [], []
    last_lower, last_upper = log_u2[0], None

    for m, u, lu in zip(mass[1:], u2[1:], log_u2[1:]):
        if last_upper is None:
            if lu - last_lower > 1.0:
                upper_m.append(m); upper_u2.append(u)
                last_upper = lu
            else:
                lower_m.append(m); lower_u2.append(u)
                last_lower = lu
        elif abs(lu - last_lower) <= abs(lu - last_upper):
            lower_m.append(m); lower_u2.append(u)
            last_lower = lu
        else:
            upper_m.append(m); upper_u2.append(u)
            last_upper = lu

    return ((np.array(lower_m), np.array(lower_u2)),
            (np.array(upper_m), np.array(upper_u2)))


def load_digitized_figure3(csv_dir: str = FIGURE3_CSV_DIR) -> dict:
    """
    Load and disentangle the digitized reference curves from Fig. 3 of
    arXiv:1811.00930, from SHiP_HNL_<flavor>_mixing_<variant>.csv (flavor in
    e/mu/tau, variant with/without Bc production -- the paper's own solid
    [f(b->Bc)=2.6e-3] vs. dash-dot [f(b->Bc)=0] curve pair per flavor,
    kept separate here rather than collapsed into one envelope).

    Returns {flavor: {"with_bc": {mass_lower, u2_lower, mass_upper, u2_upper},
                       "without_bc": {...}}} for flavor in ("e", "mu", "tau").
    """
    result = {}
    for flavor in ("e", "mu", "tau"):
        result[flavor] = {}
        for key, suffix in FIGURE3_CSV_VARIANTS.items():
            path = os.path.join(csv_dir, f"SHiP_HNL_{flavor}_mixing_{suffix}.csv")
            raw = np.loadtxt(path, delimiter=",")
            mass, u2 = raw[:, 0], raw[:, 1]
            (lm, lu2), (um, uu2) = _split_cigar_branches(mass, u2)
            result[flavor][key] = {
                "mass_lower": lm, "u2_lower": lu2,
                "mass_upper": um, "u2_upper": uu2,
            }
    return result


def plot_sensitivity_curve(data: dict, n_pot: float, pot: float, outfile: str,
                            det_efficiency: float = 1.0, target_n_events: float = 2.3,
                            u2_bounds=(1e-12, 1.0), n_scan: int = 300,
                            overlay_figure3: bool = True) -> None:
    """
    Reproduction of Fig. 3 of arXiv:1811.00930: 90% CL sensitivity curves
    for HNLs mixing to a single SM flavour (e, mu, tau), as closed "cigar"
    contours in the (m_N, U_alpha^2) plane. See sensitivity_curve for the
    boundary construction; eps_det is not modeled (flat multiplier, default
    1.0), and only the charged mesons D+, Ds+, B+, Bc+ feed production (see
    plot_production_fraction_vs_mass), so this is not expected to land on
    the paper's absolute scale -- see the known ~260x production-rate
    normalization gap flagged earlier.

    If overlay_figure3, also draws the digitized reference curves from the
    paper's actual Fig. 3 (load_digitized_figure3) as thin lines in the same
    per-flavor colors -- dotted for the with-Bc-production variant (the
    paper's solid curve, f(b->Bc)=2.6e-3) and dash-dot for the without-Bc
    variant (the paper's dash-dot curve, f(b->Bc)=0) -- for direct visual
    comparison.

    Below a certain mass (set by how steeply Gamma_N falls with M_N), the
    HNL never becomes short-lived enough to decay before the far edge of
    the decay volume even at the unphysical U_alpha^2 = 1, so the "too
    short-lived" upper boundary doesn't exist within u2_bounds -- only the
    lower (rare-decay) boundary does. Rather than silently dropping these
    masses, that open-ended region is drawn as a lightly-hatched band
    running from the lower boundary up to u2_bounds[1], bounded by a
    dashed (not solid) line on top to mark that the true sensitive region
    actually extends further (to U_alpha^2 > u2_bounds[1], off-scale/
    unphysical) rather than closing there.
    """
    colors = {"e": "#0C5DA5", "mu": "#FF2C00", "tau": "#00B945"}
    labels = {"e": r"$\alpha=e$", "mu": r"$\alpha=\mu$", "tau": r"$\alpha=\tau$"}

    fig, ax = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)

    drew_open_ended = False
    for flavor in ("e", "mu", "tau"):
        masses, u2_lower, u2_upper = sensitivity_curve(
            data, n_pot, pot, flavor, det_efficiency, target_n_events, u2_bounds, n_scan)
        closed = ~np.isnan(u2_lower) & ~np.isnan(u2_upper)
        open_ended = ~np.isnan(u2_lower) & np.isnan(u2_upper)

        if np.any(closed):
            ax.fill_between(masses[closed], u2_lower[closed], u2_upper[closed],
                             color=colors[flavor], alpha=0.25, label=labels[flavor])
            ax.plot(masses[closed], u2_lower[closed], color=colors[flavor], lw=3)
            ax.plot(masses[closed], u2_upper[closed], color=colors[flavor], lw=3)

        if np.any(open_ended):
            drew_open_ended = True
            label = labels[flavor] if not np.any(closed) else None
            ax.fill_between(masses[open_ended], u2_lower[open_ended], u2_bounds[1],
                             color=colors[flavor], alpha=0.12, hatch="//",
                             edgecolor=colors[flavor], linewidth=0.0, label=label)
            ax.plot(masses[open_ended], u2_lower[open_ended], color=colors[flavor], lw=3)
            ax.plot(masses[open_ended], np.full(np.sum(open_ended), u2_bounds[1]),
                     color=colors[flavor], lw=1.5, ls="--")

    if drew_open_ended:
        ax.plot([], [], color="black", lw=1.5, ls="--",
                 label=rf"sensitive region extends past $U_\alpha^2={u2_bounds[1]:g}$ (off-scale)")

    figure3_csv_paths = [
        os.path.join(FIGURE3_CSV_DIR, f"SHiP_HNL_{flavor}_mixing_{suffix}.csv")
        for flavor in ("e", "mu", "tau") for suffix in FIGURE3_CSV_VARIANTS.values()
    ]
    if overlay_figure3 and all(os.path.exists(p) for p in figure3_csv_paths):
        digitized = load_digitized_figure3()
        variant_style = {"with_bc": ":", "without_bc": "-."}
        for flavor in ("e", "mu", "tau"):
            for variant, ls in variant_style.items():
                curve = digitized[flavor][variant]
                ax.plot(curve["mass_lower"], curve["u2_lower"], color=colors[flavor], lw=1.5, ls=ls)
                ax.plot(curve["mass_upper"], curve["u2_upper"], color=colors[flavor], lw=1.5, ls=ls)
        # dummy handles for the legend, since the per-flavor colors are
        # already explained by the filled regions above
        ax.plot([], [], color="black", lw=1.5, ls=":",
                 label=r"arXiv:1811.00930 Fig. 3 (digitized, with $b\to B_c$)")
        ax.plot([], [], color="black", lw=1.5, ls="-.",
                 label=r"arXiv:1811.00930 Fig. 3 (digitized, without $b\to B_c$)")

    ax.set_xlabel(r"HNL mass $M_N$ [GeV]")
    ax.set_ylabel(r"$U_\alpha^2$")
    ax.set_yscale("log")
    ax.set_title(rf"{{\bf SHiP 90\% CL sensitivity}} ($\bar N_{{\rm events}} \geq {target_n_events:g}$)")
    legend_if_any(ax, loc="upper right")
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.45)

    fig.savefig(outfile, dpi=200)


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
    pz_N = data["pz_N"][in_bin]  # GeV, needed below for the proper-time calc
    vz = data["vz"][in_bin]  # mm, needed below for the proper-time calc
    vx_DV, vy_DV, forward_going = _propagate_to_decay_volume(data, in_bin)

    # apply geometrical cut ('and'/'or' don't broadcast over numpy arrays;
    # need the elementwise '&' operator here). forward_going is required
    # first: for pz_N <= 0 the extrapolated position above is not
    # physically meaningful (a backward/transverse-going HNL can never
    # reach a downstream decay volume), and the transverse coordinates can
    # spuriously fall inside the acceptance window by coincidence.
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


def _masked_hist2d(ax, x: np.ndarray, y: np.ndarray, bins: int, hist_range: list):
    """
    2D histogram with zero-count bins masked out, so they render as
    transparent (background shows through) rather than being painted by
    the colormap. Plain ax.hist2d(..., norm=LogNorm()) looks fine when the
    populated bins span a range of counts, but when every populated bin
    happens to have the *same* count (common for sparse data and/or fine
    binning, e.g. a zoomed-in inset) LogNorm's auto-scaled vmin == vmax,
    and matplotlib fills the *entire* axes -- including empty bins -- with
    one flat color instead of leaving them blank. Masking zero bins here
    sidesteps that degenerate-normalization case entirely.
    """
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range)
    masked_counts = np.ma.masked_equal(counts, 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0)
    vmax = max(int(masked_counts.max()), 2) if masked_counts.count() else 2
    return ax.pcolormesh(xedges, yedges, masked_counts.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax))


def plot_decay_volume_xy(data: dict, target_mass: float, outfile: str, n_bins: int = 100) -> None:
    """
    2D histogram of the (x, y) position of every forward-going HNL
    (pz_N > 0) at the mass grid point closest to target_mass, extrapolated
    to the decay-volume entrance plane (z = Z_DECAY_VOLUME) via the same
    straight-line propagation used for the survival probability. Colored by
    raw simulated counts, not flux -- this is a geometric/kinematic
    diagnostic of the HNL beam's transverse profile at the decay volume,
    independent of the active-sterile mixing pattern (which only rescales
    weights, not where a given simulated HNL's trajectory points).

    Includes *all* forward-going HNLs produced at the target, not just the
    ones that land inside the decay volume -- the overlaid rectangle (the
    decay volume's actual transverse size, centered at x=y=0) is what shows
    how much of the beam the fixed-size decay volume geometrically catches.
    """
    mass_points = np.unique(data["mN"])
    closest_mass = mass_points[np.argmin(np.abs(mass_points - target_mass))]
    in_bin = data["mN"] == closest_mass

    vx_DV, vy_DV, forward_going = _propagate_to_decay_volume(data, in_bin)
    x_m = vx_DV[forward_going] / 1e3  # mm -> m
    y_m = vy_DV[forward_going] / 1e3  # mm -> m

    # Data-driven, symmetric view window: wide enough to always show the
    # decay-volume rectangle with margin, even if the simulated beam
    # footprint is tighter or much wider than the decay volume itself.
    half_x = max(2 * X_DECAY_VOLUME / 1e3, np.percentile(np.abs(x_m), 99) if x_m.size else 0.0)
    half_y = max(2 * Y_DECAY_VOLUME / 1e3, np.percentile(np.abs(y_m), 99) if y_m.size else 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)

    image = _masked_hist2d(ax, x_m, y_m, n_bins, [[-half_x, half_x], [-half_y, half_y]])

    ax.add_patch(Rectangle(
        (-X_DECAY_VOLUME / 1e3, -Y_DECAY_VOLUME / 1e3),
        2 * X_DECAY_VOLUME / 1e3, 2 * Y_DECAY_VOLUME / 1e3,
        fill=False, edgecolor="#FF2C00", linewidth=3, linestyle="--",
        label="Decay volume acceptance",
    ))

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Simulated HNL counts")

    ax.set_xlabel(r"$x$ at decay volume [m]")
    ax.set_ylabel(r"$y$ at decay volume [m]")
    ax.set_title(rf"{{\bf Forward-going HNLs at decay volume}} ($M_N={closest_mass:.3g}$ GeV)")
    ax.set_aspect("equal")
    legend_if_any(ax, loc="upper right")

    # Inset: zoomed-in view of the decay-volume acceptance rectangle itself
    # (the full-range plot above is dominated by the much wider beam
    # footprint, so the rectangle -- and whatever falls inside it -- would
    # otherwise be too small to read). View window is a fixed margin around
    # the rectangle, independent of the outer plot's data-driven range.
    zoom_half_x = 1.5 * X_DECAY_VOLUME / 1e3
    zoom_half_y = 1.5 * Y_DECAY_VOLUME / 1e3

    axins = ax.inset_axes([0.60, 0.05, 0.35, 0.35])
    _masked_hist2d(axins, x_m, y_m, n_bins, [[-zoom_half_x, zoom_half_x], [-zoom_half_y, zoom_half_y]])
    axins.add_patch(Rectangle(
        (-X_DECAY_VOLUME / 1e3, -Y_DECAY_VOLUME / 1e3),
        2 * X_DECAY_VOLUME / 1e3, 2 * Y_DECAY_VOLUME / 1e3,
        fill=False, edgecolor="#FF2C00", linewidth=2, linestyle="--",
    ))
    axins.set_xlim(-zoom_half_x, zoom_half_x)
    axins.set_ylim(-zoom_half_y, zoom_half_y)
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    fig.savefig(outfile, dpi=200)


def write_summary_file(outfile: str, data: dict, target_mass: float, closest_mass: float,
                        Ue2: float, Umu2: float, Utau2: float,
                        n_pot: float, pot: float, det_efficiency: float,
                        bins: np.ndarray, spectra: dict, spectra_DV: dict) -> None:
    """
    Write a text summary of the run parameters (HNL mass, mixing angles,
    decay volume geometry, POT) and the resulting HNL yields, obtained by
    integrating the differential energy spectra dPhi_N/dE_N (both at the
    target and at the decay volume) over dE_N, plus the expected number of
    detected events (arXiv:1811.00930 eqs. 2.1, 2.4, 2.5 -- see
    expected_n_events).
    """
    bin_widths = np.diff(bins)
    n_per_pot_target = float(np.sum((spectra["light"] + spectra["heavy"]) * bin_widths))
    n_per_pot_DV = float(np.sum((spectra_DV["light"] + spectra_DV["heavy"]) * bin_widths))
    decay_width = total_decay_width(closest_mass, Ue2, Umu2, Utau2)
    _, n_prod, n_events = expected_n_events(data, n_pot, target_mass, Ue2, Umu2, Utau2, pot, det_efficiency)
    br_visible = 1.0 - invisible_decay_width(closest_mass, Ue2, Umu2, Utau2) / decay_width

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
        f.write(f"Length [m]:               {DECAY_VOLUME_LENGTH / 1e3:.6g}\n")
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
        f.write(f"Total HNLs detected at decay volume for {pot:.6g} POT: {n_per_pot_DV * pot:.6g}\n\n")

        f.write("Expected number of detected events (eqs. 2.1, 2.4, 2.5 of arXiv:1811.00930)\n")
        f.write("------------------------------------------------------------------------------\n")
        f.write("NOTE: Pdet = Pdecay * BR(N->visible) * eps_det, with eps_det (detection/\n")
        f.write("reconstruction efficiency) fixed at the flat value below -- not yet modeled.\n")
        f.write(f"Detection efficiency eps_det (flat, not modeled): {det_efficiency:.6g}\n")
        f.write(f"BR(N -> visible) = 1 - Gamma_invisible/Gamma_N:   {br_visible:.6g}\n")
        f.write(f"N_prod (produced, geometrically accepted) for {pot:.6g} POT: {n_prod:.6g}\n")
        f.write(f"N_events (expected detected) for {pot:.6g} POT:             {n_events:.6g}\n")


def load_macro_file(path: str) -> dict:
    """
    Parse a macro file of 'key = value' lines into a dict of raw string
    values (blank lines and '#' comments -- full-line or trailing -- are
    ignored). Keys match the long-form CLI argument names in main() below
    (e.g. 'mass', 'Umu2', 'decay_volume_half_width'), so a macro file is a
    saveable, shareable alternative to a long command line: see
    example_run.mac for a template. Any parameter also given explicitly on
    the command line overrides the macro file's value for that parameter
    (see the `resolve` helper in main()).
    """
    params = {}
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{lineno}: expected 'key = value', got: {raw_line!r}")
            key, value = line.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def _parse_macro_bool(value: str) -> bool:
    if value.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if value.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Expected a boolean (true/false/yes/no/1/0), got: {value!r}")


def _run_if_missing(outfile: str, remake: bool, fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) (which must save its figure to outfile) unless
    outfile already exists and remake is False, in which case skip with a
    notice. Intended for plots that don't depend on the active-sterile
    mixing pattern (Ue2/Umu2/Utau2) -- plot_production_fraction_vs_mass,
    plot_decay_volume_xy, plot_sensitivity_curve -- so that runs which only
    vary the mixing pattern don't needlessly regenerate identical plots.
    """
    if not remake and os.path.exists(outfile):
        print(f"[skip] {outfile} already exists (pass --remake-static-plots to regenerate)")
        return None
    return fn(*args, **kwargs)


def main():
    # Declared up front: the --decay-volume-* help strings below read the
    # current module-level defaults, and Python requires `global` to appear
    # before any use of the name in this scope.
    global Z_DECAY_VOLUME, X_DECAY_VOLUME, Y_DECAY_VOLUME, DECAY_VOLUME_LENGTH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=str, default=None,
                         help="Path to a macro file of 'key = value' lines providing run "
                              "parameters (see example_run.mac). Any of the flags below, "
                              "if also given explicitly, overrides that parameter's value "
                              "from the macro file.")
    parser.add_argument("--root", default=argparse.SUPPRESS,
                         help="Path to the Geant4 output ROOT file.")
    parser.add_argument("--npot", default=argparse.SUPPRESS,
                         help="Path to the companion protons-on-target count.")
    parser.add_argument("--mass", type=float, default=argparse.SUPPRESS,
                         help="HNL mass [GeV] for the energy-spectrum plot "
                              "(snaps to the nearest simulated benchmark mass).")
    parser.add_argument("--Ue2", type=float, default=argparse.SUPPRESS, help="|U_e|^2 mixing.")
    parser.add_argument("--Umu2", type=float, default=argparse.SUPPRESS, help="|U_mu|^2 mixing.")
    parser.add_argument("--Utau2", type=float, default=argparse.SUPPRESS, help="|U_tau|^2 mixing.")
    parser.add_argument("--pot", type=float, default=argparse.SUPPRESS,
                         help="Protons-on-target to scale the total HNL yield by "
                              "(defaults to the simulated POT count from --npot, "
                              "i.e. no additional scaling).")
    parser.add_argument("--decay-volume-z", type=float, default=argparse.SUPPRESS,
                         help="Distance from the target (z=0) to the decay volume [m] "
                              f"(default: {Z_DECAY_VOLUME / 1e3:g}).")
    parser.add_argument("--decay-volume-half-width", type=float, default=argparse.SUPPRESS,
                         help="Decay volume half-width in x [m] "
                              f"(default: {X_DECAY_VOLUME / 1e3:g}).")
    parser.add_argument("--decay-volume-half-height", type=float, default=argparse.SUPPRESS,
                         help="Decay volume half-height in y [m] "
                              f"(default: {Y_DECAY_VOLUME / 1e3:g}).")
    parser.add_argument("--decay-volume-length", type=float, default=argparse.SUPPRESS,
                         help="Decay volume length along z [m], entrance to exit "
                              f"(default: {DECAY_VOLUME_LENGTH / 1e3:g}).")
    parser.add_argument("--det-efficiency", type=float, default=argparse.SUPPRESS,
                         help="Detection efficiency eps_det (track reconstruction + "
                              "selection), applied as a flat multiplier to N_events "
                              "(default: 1.0, i.e. not modeled).")
    parser.add_argument("--outdir", default=argparse.SUPPRESS, help="Output directory for plots.")
    parser.add_argument("--breakdown", action="store_true", default=argparse.SUPPRESS,
                         help="Plot each parent meson species individually instead of "
                              "the aggregated light (pi/K) / heavy (D/Ds/B/Bc) curves.")
    parser.add_argument("--remake-static-plots", action="store_true", default=argparse.SUPPRESS,
                         help="Force regeneration of plots that don't depend on the mixing "
                              "pattern (production fraction, decay-volume xy, sensitivity "
                              "curve) even if their output file already exists. By default "
                              "these are skipped when present, since runs that only vary "
                              "Ue2/Umu2/Utau2 would otherwise regenerate identical plots.")
    parser.add_argument("--tag", type=str, default=argparse.SUPPRESS, help="Tag to append at end of figure names")
    args = parser.parse_args()
    args_dict = vars(args)

    macro = load_macro_file(args.macro) if args.macro else {}

    def resolve(name, cast, default):
        # CLI (only present if explicitly passed, via default=SUPPRESS)
        # beats the macro file, which beats the hardcoded default.
        if name in args_dict:
            return args_dict[name]
        if name in macro:
            return cast(macro[name])
        return default

    root = resolve("root", str, "../geant4/build/HNL_target_flux.root")
    npot_path = resolve("npot", str, "../geant4/build/n_pot.txt")
    mass = resolve("mass", float, 1.0)
    Ue2 = resolve("Ue2", float, 0.0)
    Umu2 = resolve("Umu2", float, 1.0)
    Utau2 = resolve("Utau2", float, 0.0)
    pot_override = resolve("pot", float, None)
    dv_z = resolve("decay_volume_z", float, None)
    dv_half_width = resolve("decay_volume_half_width", float, None)
    dv_half_height = resolve("decay_volume_half_height", float, None)
    dv_length = resolve("decay_volume_length", float, None)
    det_efficiency = resolve("det_efficiency", float, 1.0)
    outdir = resolve("outdir", str, "../plots")
    breakdown = resolve("breakdown", _parse_macro_bool, False)
    remake_static = resolve("remake_static_plots", _parse_macro_bool, False)
    tag = resolve("tag", str, None)

    # Decay volume geometry is used as module-level constants throughout
    # (energy_spectrum_table, _propagate_to_decay_volume,
    # plot_decay_volume_xy, write_summary_file); override them here, before
    # any of those run, if the macro file/CLI asked for non-default geometry.
    if dv_z is not None:
        Z_DECAY_VOLUME = dv_z * 1e3
    if dv_half_width is not None:
        X_DECAY_VOLUME = dv_half_width * 1e3
    if dv_half_height is not None:
        Y_DECAY_VOLUME = dv_half_height * 1e3
    if dv_length is not None:
        DECAY_VOLUME_LENGTH = dv_length * 1e3

    data, n_pot = load_events(root, npot_path)
    pot = pot_override if pot_override is not None else n_pot

    if tag is not None:
        plot_flux_vs_mass(data, n_pot, Ue2, Umu2, Utau2,
                          f"{outdir}/HNL_target_flux_vs_mass_{tag}.pdf", breakdown=breakdown)
        closest_mass, bins, spectra, spectra_DV = plot_energy_spectrum(
            data, n_pot, mass, Ue2, Umu2, Utau2,
            f"{outdir}/HNL_target_energy_spectrum_{tag}.pdf",
            f"{outdir}/HNL_decay_volume_energy_spectrum_{tag}.pdf",
            breakdown=breakdown)
        write_summary_file(f"{outdir}/HNL_summary_{tag}.txt", data,
                            mass, closest_mass, Ue2, Umu2, Utau2,
                            n_pot, pot, det_efficiency, bins, spectra, spectra_DV)
        _run_if_missing(f"{outdir}/HNL_decay_volume_xy_{tag}.pdf", remake_static,
                         plot_decay_volume_xy, data, mass, f"{outdir}/HNL_decay_volume_xy_{tag}.pdf")
        _run_if_missing(f"{outdir}/HNL_production_fraction_{tag}.pdf", remake_static,
                         plot_production_fraction_vs_mass, data, n_pot,
                         f"{outdir}/HNL_production_fraction_{tag}.pdf")
        _run_if_missing(f"{outdir}/HNL_sensitivity_{tag}.pdf", remake_static,
                         plot_sensitivity_curve, data, n_pot, pot, f"{outdir}/HNL_sensitivity_{tag}.pdf",
                         det_efficiency=det_efficiency)
    else:
        plot_flux_vs_mass(data, n_pot, Ue2, Umu2, Utau2,
                          f"{outdir}/HNL_target_flux_vs_mass.pdf", breakdown=breakdown)
        closest_mass, bins, spectra, spectra_DV = plot_energy_spectrum(
            data, n_pot, mass, Ue2, Umu2, Utau2,
            f"{outdir}/HNL_target_energy_spectrum.pdf",
            f"{outdir}/HNL_decay_volume_energy_spectrum.pdf",
            breakdown=breakdown)
        write_summary_file(f"{outdir}/HNL_summary.txt", data,
                            mass, closest_mass, Ue2, Umu2, Utau2,
                            n_pot, pot, det_efficiency, bins, spectra, spectra_DV)
        _run_if_missing(f"{outdir}/HNL_decay_volume_xy.pdf", remake_static,
                         plot_decay_volume_xy, data, mass, f"{outdir}/HNL_decay_volume_xy.pdf")
        _run_if_missing(f"{outdir}/HNL_production_fraction.pdf", remake_static,
                         plot_production_fraction_vs_mass, data, n_pot,
                         f"{outdir}/HNL_production_fraction.pdf")
        _run_if_missing(f"{outdir}/HNL_sensitivity.pdf", remake_static,
                         plot_sensitivity_curve, data, n_pot, pot, f"{outdir}/HNL_sensitivity.pdf",
                         det_efficiency=det_efficiency)


if __name__ == "__main__":
    main()
