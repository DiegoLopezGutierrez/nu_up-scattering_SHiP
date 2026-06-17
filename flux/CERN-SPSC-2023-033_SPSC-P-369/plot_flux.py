import csv
import matplotlib.pyplot as plt
import numpy as np

STYLE_DIR = '../../plots/'
plt.style.use(STYLE_DIR+'sty.mplstyle')

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


# initialize fluxes
numu_numubar_flux = []
numu_numubar_energy = []

nue_nuebar_flux = []
nue_nuebar_energy = []

nutau_nutaubar_flux = []
nutau_nutaubar_energy = []

# in Fig. 56 of the 2023 report, they indicate A.U. for the y-axis.
# I will then just compare the shapes. I will include an airbitrary normalization to try to match with the other fluxes I have.
normalization_POT = 4e13

# assumed area of 40 cm x 40 cm, see Sec. 3.1.1 in 2023 report.
normalization_area = 0.40*0.40 # m^2

# based on visual inspection of Fig. 56 in the CERN SHiP 2023 report, bin width is 2 GeV
bins = np.arange(0,300,2)

def extract_data(filename : str, energy : list, flux : list, normalization : float = 1.0) -> None:
    with open(filename,'r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',') 
        for row in data:
            energy.append(float(row[0]))
            flux.append(float(row[1]) / normalization)

# retrieve fluxes
extract_data('SHiP_2023Report_numu_numubar_flux.csv', numu_numubar_energy, numu_numubar_flux)
extract_data('SHiP_2023Report_nue_nuebar_flux.csv', nue_nuebar_energy, nue_nuebar_flux)
extract_data('SHiP_2023Report_nutau_nutaubar_flux.csv', nutau_nutaubar_energy, nutau_nutaubar_flux)

# produce histograms
numu_numubar_hist, numu_numubar_edges = np.histogram(numu_numubar_energy, bins=bins, weights=numu_numubar_flux)
nue_nuebar_hist, nue_nuebar_edges = np.histogram(nue_nuebar_energy, bins=bins, weights=nue_nuebar_flux)
nutau_nutaubar_hist, nutau_nutaubar_edges = np.histogram(nutau_nutaubar_energy, bins=bins, weights=nutau_nutaubar_flux)

numu_numubar_centers = edges_to_centers(numu_numubar_edges)
nue_nuebar_centers = edges_to_centers(nue_nuebar_edges)
nutau_nutaubar_centers = edges_to_centers(nutau_nutaubar_edges)

log_bins    = np.geomspace(1,400,num=41,endpoint=True)
log_centers = edges_to_centers(log_bins)

numu_numubar_rebinned = rebin_counts(numu_numubar_hist, numu_numubar_edges, log_bins)
nue_nuebar_rebinned = rebin_counts(nue_nuebar_hist, nue_nuebar_edges, log_bins)
nutau_nutaubar_rebinned = rebin_counts(nutau_nutaubar_hist, nutau_nutaubar_edges, log_bins)

# plot (old) retrieved fluxes in linear bins
fig, ax = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)

cols = ['purple',
        'orange',
        'teal']

ax.hist(numu_numubar_centers, density=False, bins=numu_numubar_edges, weights=numu_numubar_hist, histtype='step', color=cols[0], label=r'$\nu_\mu+\bar{\nu}_\mu$', alpha=1, lw=5)
ax.hist(nue_nuebar_centers, density=False, bins=nue_nuebar_edges, weights=nue_nuebar_hist, histtype='step', color=cols[1], label=r'$\nu_e+\bar{\nu}_e$', alpha=1, lw=5)
ax.hist(nutau_nutaubar_centers, density=False, bins=nutau_nutaubar_edges, weights=nutau_nutaubar_hist, histtype='step', color=cols[2], label=r'$\nu_\tau+\bar{\nu}_\tau$', alpha=1, lw=5)

# styling
ax.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV/c]')
ax.set_ylabel(r'Flux $\Phi$ [$\nu/(0.40\times 0.40\,\mathrm{m}^2)/(4\times 10^{13}\,\mathrm{POT})$]')
ax.set_title(r'{\bf At detector (CERN-SPSC-2023-033/SPSC-P-369)}')
ax.set_xscale('linear')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax.set_xlim(1, 300)

fig.savefig("flux_SHiP_at_detector_linearbins.pdf", dpi=100)

# plot (new) fluxes in log bins
fig2, ax2 = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)

cols = ['purple',
        'orange',
        'teal']

ax2.hist(log_centers, density=False, bins=log_bins, weights=numu_numubar_rebinned/normalization_POT/normalization_area, histtype='step', color=cols[0], label=r'$\nu_\mu+\bar{\nu}_\mu$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=nue_nuebar_rebinned/normalization_POT/normalization_area, histtype='step', color=cols[1], label=r'$\nu_e+\bar{\nu}_e$', alpha=1, lw=5)
ax2.hist(log_centers, density=False, bins=log_bins, weights=nutau_nutaubar_rebinned/normalization_POT/normalization_area, histtype='step', color=cols[2], label=r'$\nu_\tau+\bar{\nu}_\tau$', alpha=1, lw=5)

# styling
ax2.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV/c]')
ax2.set_ylabel(r'Flux $\Phi$ [$\nu/\mathrm{m}^2/\mathrm{POT}$]')
ax2.set_title(r'{\bf At detector (CERN-SPSC-2023-033/SPSC-P-369)}')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='upper right')
ax2.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax2.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax2.set_xlim(1, 400)

fig2.savefig("flux_SHiP_at_detector_logbins.pdf", dpi=100)