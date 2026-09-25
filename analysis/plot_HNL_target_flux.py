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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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

    def diff_flux(mask: np.ndarray) -> np.ndarray:
        # np.histogram gives the flux *integrated within each bin* (N/POT);
        # dividing by the bin width turns that into a true differential
        # quantity dPhi/dE_N (N/GeV/POT) -- necessary here because the bins
        # are log-spaced (width grows with energy), so skipping this step
        # would distort the spectrum shape, not just its overall scale.
        counts, _ = np.histogram(energy[mask], bins=bins, weights=weight[mask])
        return counts / bin_widths

    spectra = {pdg: diff_flux(parent == pdg) for pdg in PARENT_LABEL}
    spectra["light"] = diff_flux(np.isin(parent, LIGHT_PARENTS))
    spectra["heavy"] = diff_flux(np.isin(parent, HEAVY_PARENTS))

    return closest_mass, bins, spectra


def plot_energy_spectrum(data: dict, n_pot: float, target_mass: float,
                          Ue2: float, Umu2: float, Utau2: float, outfile: str,
                          breakdown: bool = False) -> None:
    """
    Differential HNL flux vs. lab energy at the mass grid point closest to
    target_mass, split into light-meson (pi/K) and heavy-flavor (D/Ds/B/Bc)
    contributions (or into individual meson species if breakdown=True).
    """
    closest_mass, bins, spectra = energy_spectrum_table(data, n_pot, target_mass, Ue2, Umu2, Utau2)

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
    parser.add_argument("--outdir", default="../plots", help="Output directory for plots.")
    parser.add_argument("--breakdown", action="store_true",
                         help="Plot each parent meson species individually instead of "
                              "the aggregated light (pi/K) / heavy (D/Ds/B/Bc) curves.")
    parser.add_argument("--tag", type=str, default=None, help="Tag to append at end of figure names")
    args = parser.parse_args()

    data, n_pot = load_events(args.root, args.npot)

    if args.tag is not None:
        plot_flux_vs_mass(data, n_pot, args.Ue2, args.Umu2, args.Utau2,
                          f"{args.outdir}/HNL_target_flux_vs_mass_{args.tag}.pdf", breakdown=args.breakdown)
        plot_energy_spectrum(data, n_pot, args.mass, args.Ue2, args.Umu2, args.Utau2,
                             f"{args.outdir}/HNL_target_energy_spectrum_{args.tag}.pdf", breakdown=args.breakdown)
    else:
        plot_flux_vs_mass(data, n_pot, args.Ue2, args.Umu2, args.Utau2,
                          f"{args.outdir}/HNL_target_flux_vs_mass.pdf", breakdown=args.breakdown)
        plot_energy_spectrum(data, n_pot, args.mass, args.Ue2, args.Umu2, args.Utau2,
                             f"{args.outdir}/HNL_target_energy_spectrum.pdf", breakdown=args.breakdown)


if __name__ == "__main__":
    main()
