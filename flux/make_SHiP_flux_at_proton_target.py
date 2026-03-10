import ROOT
import matplotlib.pyplot as plt
import numpy as np
import csv

STYLE_DIR = '../plots/'
plt.style.use(STYLE_DIR+'sty.mplstyle')

PYTHIA_FILE = 'pythia8_Geant4_1.0_withCharm_nu.root'
root_file = ROOT.TFile.Open(PYTHIA_FILE)

## helper functions
def centers_to_edges(centers) -> np.ndarray:
    """
    Calculate the bin edges given the bin centers.
    """
    centers = np.asarray(centers)
    # Interior edges = midpoints
    edges = (centers[:-1] + centers[1:]) / 2
    # Extrapolate first and last edges
    first = centers[0] - (centers[1] - centers[0]) / 2
    last  = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[first], edges, [last]])

def edges_to_centers(edges) -> np.ndarray:
    """
    Calculate the bin centers given the bin edges.
    """
    edges = np.asarray(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers

def rebin_counts(counts_old: np.ndarray, edges_old: np.ndarray, edges_new: np.ndarray) -> np.ndarray:
    """
    Calculate the new bin counts for new bins (edges_new) given the old counts (counts_old) and the old bins (edges_old).
    This function assumes a uniform count distribution within each old bin. Given this assumption, the old count
    contribution to the new count is proportional to the fractional overlap of the old bin with the new bin.
    """
    counts_new = np.zeros(len(edges_new)-1)
    for i in range(len(edges_new)-1):
        a, b = edges_new[i], edges_new[i+1]
        for j in range(len(edges_old)-1):
            c, d = edges_old[j], edges_old[j+1]
            left  = max(a, c)
            right = min(b, d)
            if right > left:
                frac = (right - left) / (d - c)
                counts_new[i] += frac * counts_old[j]
    return counts_new

def save_flux(counts: np.ndarray, edges: np.ndarray, filename: str, top_row: list | None = None) -> None:
    """
    Save the flux given some bin edges (edges) and an array of bin counts (counts).
    """
    with open(filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if top_row is not None:
            writer.writerow(top_row)
        for i in range(len(edges)-1):
            writer.writerow([edges[i], edges[i+1]] + list(counts[:,i]))

def make_diff_flux(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Calculate the differential neutrino flux with respect to energy at each bin given 
    the incoming neutrinos (counts) and its energy bins (edges).
    """
    diff_flux = np.zeros(len(counts))
    bin_widths = np.zeros(len(counts))
    for i in range(len(counts)):
        dE = edges[i+1] - edges[i]
        count = counts[i]
        diff_flux[i] = count / dE
        bin_widths[i] = dE
    return diff_flux, bin_widths

# format is in counts of neutrinos per 5e13 POT [1] vs neutrino momenta [GeV/c]
pythia_vmu     = root_file.Get("1014")  # numu flux
pythia_vmubar  = root_file.Get("2014")  # numubar flux
pythia_ve      = root_file.Get("1012")  # nue flux
pythia_vebar   = root_file.Get("2012")  # nuebar flux
pythia_vtau    = root_file.Get("1016")  # nutau flux
pythia_vtaubar = root_file.Get("2016")  # nutaubar flux

# Calculate overall normalization
NORMALIZATION = 5e13 # NPOT used for calculating this flux

# Convert ROOT histogram to NumPy arrays
bin_centers = []

# Neutrino flux at proton target
bin_contents_vmu     = []
bin_contents_vmubar  = []
bin_contents_ve      = []
bin_contents_vebar   = []
bin_contents_vtau    = []
bin_contents_vtaubar = []

for i in range(1, pythia_vmu.GetNbinsX() + 1):
    # Bin centers (neutrino energy [GeV/c])
    bin_centers.append(pythia_vmu.GetBinCenter(i))

    # bin contents (neutrino count normalized to 1 POT)
    bin_contents_vmu.append(pythia_vmu.GetBinContent(i) / NORMALIZATION)
    bin_contents_vmubar.append(pythia_vmubar.GetBinContent(i) / NORMALIZATION)
    bin_contents_ve.append(pythia_ve.GetBinContent(i) / NORMALIZATION)
    bin_contents_vebar.append(pythia_vebar.GetBinContent(i) / NORMALIZATION)
    bin_contents_vtau.append(pythia_vtau.GetBinContent(i) / NORMALIZATION)
    bin_contents_vtaubar.append(pythia_vtaubar.GetBinContent(i) / NORMALIZATION)

# initialize as arrays
bin_centers          = np.array(bin_centers)
bin_contents_vmu     = np.array(bin_contents_vmu)
bin_contents_ve      = np.array(bin_contents_ve)
bin_contents_vtau    = np.array(bin_contents_vtau)
bin_contents_vmubar  = np.array(bin_contents_vmubar)
bin_contents_vebar   = np.array(bin_contents_vebar)
bin_contents_vtaubar = np.array(bin_contents_vtaubar)

# calculate total neutrino count per POT for future test
total_count_vmu      = np.sum(bin_contents_vmu)
total_count_ve       = np.sum(bin_contents_ve)
total_count_vtau     = np.sum(bin_contents_vtau)
total_count_vmubar   = np.sum(bin_contents_vmubar)
total_count_vebar    = np.sum(bin_contents_vebar)
total_count_vtaubar  = np.sum(bin_contents_vtaubar)

# get the bin edges for the linear (old) and logarithmic (new) bins
linear_bins = centers_to_edges(bin_centers)
log_bins    = np.geomspace(1,400,num=41,endpoint=True)
log_centers = edges_to_centers(log_bins)

# get new bin counts for the logarithmic bins
rebinned_contents_vmu     = rebin_counts(bin_contents_vmu,linear_bins,log_bins)
rebinned_contents_ve      = rebin_counts(bin_contents_ve,linear_bins,log_bins)
rebinned_contents_vtau    = rebin_counts(bin_contents_vtau,linear_bins,log_bins)
rebinned_contents_vmubar  = rebin_counts(bin_contents_vmubar,linear_bins,log_bins)
rebinned_contents_vebar   = rebin_counts(bin_contents_vebar,linear_bins,log_bins)
rebinned_contents_vtaubar = rebin_counts(bin_contents_vtaubar,linear_bins,log_bins)

# save flux
rebinned_counts = np.array([rebinned_contents_vmu, 
                            rebinned_contents_vmubar, 
                            rebinned_contents_ve, 
                            rebinned_contents_vebar,
                            rebinned_contents_vtau,
                            rebinned_contents_vtaubar])

top_row = ['Elow [GeV]',
           ' Ehigh [GeV]',
           ' numu [POT^-1]',
           ' numubar [POT^-1]',
           ' nue [POT^-1]',
           ' nuebar [POT^-1]',
           ' nutau [POT^-1]',
           ' nutaubar [POT^-1]']

save_flux(rebinned_counts,log_bins,'normalized_flux_at_proton_target.csv',top_row=top_row)

# make the differential flux
rebinned_contents_diff_vmu, log_widths = make_diff_flux(rebinned_contents_vmu, log_bins)
rebinned_contents_diff_ve, _           = make_diff_flux(rebinned_contents_ve, log_bins)
rebinned_contents_diff_vtau, _         = make_diff_flux(rebinned_contents_vtau, log_bins)
rebinned_contents_diff_vmubar, _       = make_diff_flux(rebinned_contents_vmubar, log_bins)
rebinned_contents_diff_vebar, _        = make_diff_flux(rebinned_contents_vebar, log_bins)
rebinned_contents_diff_vtaubar, _      = make_diff_flux(rebinned_contents_vtaubar, log_bins)

assert np.abs(total_count_vmu - np.sum(np.multiply(rebinned_contents_diff_vmu, log_widths))) < 0.0001, f"Total numu neutrino counts don't agree: {total_count_vmu} != {np.sum(np.multiply(rebinned_contents_diff_vmu, log_widths))}"
assert np.abs(total_count_ve - np.sum(np.multiply(rebinned_contents_diff_ve, log_widths))) < 0.0001, f"Total nue neutrino counts don't agree: {total_count_ve} != {np.sum(np.multiply(rebinned_contents_diff_ve, log_widths))}"
assert np.abs(total_count_vtau - np.sum(np.multiply(rebinned_contents_diff_vtau, log_widths))) < 0.0001, f"Total nue neutrino counts don't agree: {total_count_vtau} != {np.sum(np.multiply(rebinned_contents_diff_vtau, log_widths))}"
assert np.abs(total_count_vmubar - np.sum(np.multiply(rebinned_contents_diff_vmubar, log_widths))) < 0.0001, f"Total numu neutrino counts don't agree: {total_count_vmubar} != {np.sum(np.multiply(rebinned_contents_diff_vmubar, log_widths))}"
assert np.abs(total_count_vebar - np.sum(np.multiply(rebinned_contents_diff_vebar, log_widths))) < 0.0001, f"Total nue neutrino counts don't agree: {total_count_vebar} != {np.sum(np.multiply(rebinned_contents_diff_vebar, log_widths))}"
assert np.abs(total_count_vtaubar - np.sum(np.multiply(rebinned_contents_diff_vtaubar, log_widths))) < 0.0001, f"Total nue neutrino counts don't agree: {total_count_vtaubar} != {np.sum(np.multiply(rebinned_contents_diff_vtaubar, log_widths))}"

# save differential flux
rebinned_diff_counts = np.array([rebinned_contents_diff_vmu,
                                 rebinned_contents_diff_vmubar, 
                                 rebinned_contents_diff_ve,
                                 rebinned_contents_diff_vebar,
                                 rebinned_contents_diff_vtau,
                                 rebinned_contents_diff_vtaubar])

top_row = ['Elow [GeV]',
           ' Ehigh [GeV]',
           ' numu [GeV^-1 POT^-1]',
           ' numubar [GeV^-1 POT^-1]',
           ' nue [GeV^-1 POT^-1]',
           ' nuebar [GeV^-1 POT^-1]',
           ' nutau [GeV^-1 POT^-1]',
           ' nutaubar [GeV^-1 POT^-1]']

save_flux(rebinned_diff_counts,log_bins,'normalized_diff_flux_at_proton_target.csv',top_row=top_row)

# plot SHiP neutrino flux at proton target
fig, ax = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)

cols = ['purple',
        'orange',
        'teal']

ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_vmu, histtype='step', color=cols[0], label=r'$\nu_\mu$', alpha=1, lw=5)
ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_ve, histtype='step', color=cols[1], label=r'$\nu_e$', alpha=1, lw=5)
ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_vtau, histtype='step', color=cols[2], label=r'$\nu_\tau$', alpha=1, lw=5)
ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_vmubar, linestyle='dashed', histtype='step', color=cols[0], label=r'$\bar{\nu}_\mu$', alpha=1, lw=5)
ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_vebar, linestyle='dashed', histtype='step', color=cols[1], label=r'$\bar{\nu}_e$', alpha=1, lw=5)
ax.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_vtaubar, linestyle='dashed', histtype='step', color=cols[2], label=r'$\bar{\nu}_\tau$', alpha=1, lw=5)

# styling
ax.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV/c]')
ax.set_ylabel(r'Flux $\Phi$ [$\nu$ / POT]')
ax.set_title(r'{\bf At proton target}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax.set_xlim(1, 400)

fig.savefig("../plots/flux_SHiP_at_proton_target.pdf", dpi=100)

# plot differential neutrino flux at SHiP
fig2, ax2 = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)

ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_vmu, histtype='step', color=cols[0], label=r'$\nu_\mu$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_ve, histtype='step', color=cols[1], label=r'$\nu_e$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_vtau, histtype='step', color=cols[2], label=r'$\nu_\tau$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_vmubar, linestyle='dashed', histtype='step', color=cols[0], label=r'$\bar{\nu}_\mu$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_vebar, linestyle='dashed', histtype='step', color=cols[1], label=r'$\bar{\nu}_e$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=rebinned_contents_diff_vtaubar, linestyle='dashed', histtype='step', color=cols[2], label=r'$\bar{\nu}_\tau$', alpha=1, lw=5)

# styling
ax2.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV/c]')
ax2.set_ylabel(r'Differential flux $\frac{\mathrm{d}\Phi}{\mathrm{d}E_\nu}$ [$\nu$ / GeV / POT]')
ax2.set_title(r'{\bf At proton target}')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='upper right')
ax2.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax2.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax2.set_xlim(1, 400)

fig2.savefig("../plots/diff_flux_SHiP_at_proton_target.pdf", dpi=100)