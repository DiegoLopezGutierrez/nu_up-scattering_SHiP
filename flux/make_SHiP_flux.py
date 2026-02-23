import ROOT
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib as mpl
import numpy as np
import csv

STYLE_DIR = '../plots/styles/'
plt.style.use(STYLE_DIR+'sty.mplstyle')

FLUX_DIR  = '../csv/fluxes/SHiP/'
FLUX_FILE = FLUX_DIR+'neutrinos_detector.root'
root_file = ROOT.TFile.Open(FLUX_FILE)

histogram_vmu = root_file.Get("hnu_mu")
histogram_vmubar = root_file.Get("hnu_mu_bar")
histogram_ve  = root_file.Get("hnu_e")
histogram_vebar  = root_file.Get("hnu_e_bar")

# Calculate overall normalization
NORMALIZATION_POT = 5e13        # NPOT used for calculating this flux
NORMALIZATION_AREA = 0.40*0.40  # detector transverse area in m^2
NORMALIZATION = NORMALIZATION_POT * NORMALIZATION_AREA

# Convert ROOT histogram to NumPy arrays
bin_centers = []

bin_contents_vmu = []
bin_contents_vmubar = []
bin_contents_ve = []
bin_contents_vebar = []

for i in range(1, histogram_vmu.GetNbinsX() + 1):
    # bin centers
    bin_centers.append(histogram_vmu.GetBinCenter(i))

    # bin contents
    bin_contents_vmu.append(histogram_vmu.GetBinContent(i) / NORMALIZATION)
    bin_contents_vmubar.append(histogram_vmubar.GetBinContent(i) / NORMALIZATION)
    bin_contents_ve.append(histogram_ve.GetBinContent(i) / NORMALIZATION)
    bin_contents_vebar.append(histogram_vebar.GetBinContent(i) / NORMALIZATION)

bin_centers = np.array(bin_centers)

bin_contents_vmu = np.array(bin_contents_vmu)
bin_contents_vmubar = np.array(bin_contents_vmubar)
bin_contents_ve = np.array(bin_contents_ve)
bin_contents_vebar = np.array(bin_contents_vebar)

# Export to CSV file
with open(FLUX_DIR+'SHiP.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for energy, vmu, vmubar, ve, vebar in zip(bin_centers, bin_contents_vmu, bin_contents_vmubar, bin_contents_ve, bin_contents_vebar):
        writer.writerow([energy,vmu, vmubar, ve, vebar]) # Flux is exported as v's / m^2 / POT

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15,15), tight_layout=True)

cols = ['#FFA500',
        '#8A2BE2',
        '#B22222',
        '#1E3282']

ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vmu, histtype='step', color=cols[0], label=r'$\nu_\mu$', alpha=1, lw=3)
#ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vmu, color=cols[0], label='_hidden', alpha=0.3, edgecolor='black', lw=0.5)

ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vmubar, histtype='step', color=cols[1], label=r'$\bar{\nu}_\mu$', alpha=1, lw=3)
#ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vmubar, color=cols[1], label='_hidden', alpha=0.3, edgecolor='black', lw=0.5)

ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_ve, histtype='step', color=cols[2], label=r'$\nu_e$', alpha=1, lw=3)
#ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_ve, color=cols[2], label='_hidden', alpha=0.3, edgecolor='black', lw=0.5)

ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vebar, histtype='step', color=cols[3], label=r'$\bar{\nu}_e$', alpha=1, lw=3)
#ax.hist(bin_centers, density=False, bins=len(bin_centers), weights=bin_contents_vebar, color=cols[3], label='_hidden', alpha=0.3, edgecolor='black', lw=0.5)

## Styling ##
ax.set_xlabel(r'\textbf{Neutrino Energy} $E_\nu$ (GeV)')
ax.set_xscale('linear')
ax.set_yscale('log')
ax.legend(loc='upper right')
ax.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax.set_ylabel(r'$\mathrm{d}\Phi$ ($\nu$ / m$^2$ / POT)')

ax.set_title(r"\textbf{Flux at SHiP}")

#ax.set_ylim(4e-8,1e-3)
#ax.set_xlim(0.1, 100)

fig.savefig("../plots/flux_SHiP.png", dpi=100)
