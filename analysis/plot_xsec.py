import csv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('../plots/sty.mplstyle')

# setting this flag includes coherent scattering when retrieving cross sections
coherent_flag = True
# setting this flag includes incoherent proton scattering when retrieving cross sections
proton_flag = True
# setting this flag includes incoherent neutron scattering when retrieving cross sections
neutron_flag = True

# setting this flag produces the plot of just mu e tridents cross sections
mue_tridents_flag = True

target_dict = {'tungsten':(184,74),
               'iron':(56,26),
               'argon': (40,18)}

SHiP_targets = ['tungsten', 'iron']

### extract cross sections ###
# this function assumes the cross section is in femtobarns and converts it to cm^2
def get_xsec(filename: str, factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    with open(filename,'r') as csvfile:
        energy = []
        xsec = []
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            energy.append(float(row[0]))
            xsec.append(float(row[1]) * 1e-39 * factor)
    return np.array(energy), np.array(xsec)

# this function retrieves and stores the energy and xsecs following the naming convention of the corresponding files
def get_xsecs(process: str, coherent: bool = True, proton: bool = True, neutron: bool = True, targets: list | None = ['tungsten']) -> dict:
    XSEC_DIR = '../cross_sections/'
    xsec_dict = {}
    if coherent:
        xsec_dict['coherent'] = {}
        for target in targets:
            if target == 'tungsten':
                tail = '_coh_W_xsec.csv'
            if target == 'argon':
                tail = '_coh_Ar_xsec.csv'
            if target == 'iron':
                tail = '_coh_Fe_xsec.csv'
            coherent_filename = XSEC_DIR + process + '/coherent/' + target + '/' + process + tail
            xsec_dict['coherent'][target] = get_xsec(coherent_filename)
    if proton:
        proton_filename = XSEC_DIR + process + '/nucleon/proton/' + process + '_nucleon_p_xsec.csv'
        xsec_dict['proton'] = get_xsec(proton_filename)
    if neutron:
        neutron_filename = XSEC_DIR + process + '/nucleon/neutron/' + process + '_nucleon_n_xsec.csv'
        xsec_dict['neutron'] = get_xsec(neutron_filename)
    return xsec_dict

# nue-initiated tridents
nue_e_e_dict = get_xsecs('ve_to_ve_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nue_mu_mu_dict = get_xsecs('ve_to_ve_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nue_tau_tau_dict = get_xsecs('ve_to_ve_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
numu_mu_e_dict = get_xsecs('ve_to_vmu_mu+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nutau_tau_e_dict = get_xsecs('ve_to_vtau_tau+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)

# numu-initiated tridents
nue_e_mu_dict = get_xsecs('vmu_to_ve_e+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
numu_e_e_dict = get_xsecs('vmu_to_vmu_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
numu_mu_mu_dict = get_xsecs('vmu_to_vmu_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
numu_tau_tau_dict = get_xsecs('vmu_to_vmu_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nutau_tau_mu_dict = get_xsecs('vmu_to_vtau_tau+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)

# nutau-initiated tridents
nue_e_tau_dict = get_xsecs('vtau_to_ve_e+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
numu_mu_tau_dict = get_xsecs('vtau_to_vmu_mu+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nutau_e_e_dict = get_xsecs('vtau_to_vtau_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nutau_mu_mu_dict = get_xsecs('vtau_to_vtau_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)
nutau_tau_tau_dict = get_xsecs('vtau_to_vtau_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, targets=SHiP_targets)

## plot cross sections
# gather all information in one place
color_nue_e_e       = '#4513D5'
#color_nue_mu_mu     = '#DC3220'
#color_nue_tau_tau   = '#A8CE36'
color_numu_mu_e     = '#A13FE5'
color_nutau_tau_e   = '#005AB5'

color_nue_e_mu      = '#299E14'
#color_numu_e_e      = '#532E11'
color_numu_mu_mu    = '#EF2698'
color_numu_tau_tau  = '#FFA500'
color_nutau_tau_mu  = '#64B0FC'

color_nue_e_tau     = '#482383'
color_numu_mu_tau   = '#FB8E54'
color_nutau_e_e     = '#7B4D4E'
color_nutau_mu_mu   = '#AD0E7D'
color_nutau_tau_tau = '#34CD66'

# some processes have the same xsec so set their colors equal to each other
color_nue_mu_mu = color_nutau_mu_mu
color_nue_tau_tau = color_numu_tau_tau
color_numu_e_e = color_nutau_e_e

nue_e_e_label = r"$\nu_e \to \nu_e e^+ e^-$"
nue_mu_mu_label = r"$\nu_e \to \nu_e \mu^+ \mu^-$"
nue_tau_tau_label = r"$\nu_e \to \nu_e \tau^+ \tau^-$"
numu_mu_e_label = r"$\nu_e \to \nu_\mu \mu^+ e^-$"
nutau_tau_e_label = r"$\nu_e \to \nu_\tau \tau^+ e^-$"

nue_e_mu_label = r"$\nu_\mu \to \nu_e e^+ \mu^-$"
numu_e_e_label = r"$\nu_\mu \to \nu_\mu e^+ e^-$"
numu_mu_mu_label = r"$\nu_\mu \to \nu_\mu \mu^+ \mu^-$"
numu_tau_tau_label = r"$\nu_\mu \to \nu_\mu \tau^+ \tau^-$"
nutau_tau_mu_label = r"$\nu_\mu \to \nu_\tau \tau^+ \mu^-$"

nue_e_tau_label = r"$\nu_\tau \to \nu_e e^+ \tau^-$"
numu_mu_tau_label = r"$\nu_\tau \to \nu_\mu \mu^+ \tau^-$"
nutau_e_e_label = r"$\nu_\tau \to \nu_\tau e^+ e^-$"
nutau_mu_mu_label = r"$\nu_\tau \to \nu_\tau \mu^+ \mu^-$"
nutau_tau_tau_label = r"$\nu_\tau \to \nu_\tau \tau^+ \tau^-$"

all_colors = [color_nue_e_e, color_nue_mu_mu, color_nue_tau_tau, color_numu_mu_e, color_nutau_tau_e,
              color_nue_e_mu, color_numu_e_e, color_numu_mu_mu, color_numu_tau_tau, color_nutau_tau_mu,
              color_nue_e_tau, color_numu_mu_tau,color_nutau_e_e, color_nutau_mu_mu, color_nutau_tau_tau]

all_xsec = [nue_e_e_dict, nue_mu_mu_dict, nue_tau_tau_dict, numu_mu_e_dict, nutau_tau_e_dict,
            nue_e_mu_dict, numu_e_e_dict, numu_mu_mu_dict, numu_tau_tau_dict, nutau_tau_mu_dict,
            nue_e_tau_dict, numu_mu_tau_dict, nutau_e_e_dict, nutau_mu_mu_dict, nutau_tau_tau_dict]

all_labels = [nue_e_e_label, nue_mu_mu_label, nue_tau_tau_label, numu_mu_e_label, nutau_tau_e_label,
              nue_e_mu_label, numu_e_e_label, numu_mu_mu_label, numu_tau_tau_label, nutau_tau_mu_label,
              nue_e_tau_label, numu_mu_tau_label, nutau_e_e_label, nutau_mu_mu_label, nutau_tau_tau_label]

# plot cross sections
fig, axes = plt.subplots(len(SHiP_targets), 2, figsize=(50,30), constrained_layout=True, sharex='col', sharey='row', squeeze=True)
trident_legend_lines = [Line2D([0], [0], color=c, lw=5) for c in all_colors]

for i in range(15):
    xsec_dict = all_xsec[i]
    trid_color = all_colors[i]
    trid_label = all_labels[i]
    for j, target in enumerate(SHiP_targets):
        energies = xsec_dict['coherent'][target][0]
        xsec_coh = xsec_dict['coherent'][target][1]
        energies_p, xsec_p = xsec_dict['proton']
        energies_n, xsec_n = xsec_dict['neutron']
        xsec_incoh = target_dict[target][1]*xsec_p + (target_dict[target][0]-target_dict[target][1])*xsec_n
        axes[j, 0].plot(energies, np.divide(xsec_coh, energies), color = trid_color, label=trid_label, linestyle='solid', linewidth=6)
        axes[j, 1].plot(energies_p, np.divide(xsec_incoh, energies_p), color = trid_color, label=trid_label, linestyle='dashed', linewidth=6)

        axes[j, 0].set_ylabel(r'Cross Section $\sigma/E_\nu$ [cm$^2$/GeV]', fontsize=50)
        for k in [0,1]:
            axes[j, k].grid()
            axes[j, k].set_xlim(1,400)
            axes[j, k].set_yscale('log')
            axes[j, k].set_xscale('log')
            #axes[j, k].legend(handles=trident_legend_lines, labels=all_labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=35)
            if j == (len(SHiP_targets)-1):
                axes[j, k].set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]', fontsize=50)

        if target == 'tungsten':
            axes[j, 0].text(0.5, 0.94, r'\bf{Coherent Scattering off Tungsten}', horizontalalignment='center', transform=axes[j, 0].transAxes, fontsize=35)
            axes[j, 1].text(0.5, 0.94, r'\bf{Incoherent Scattering off Tungsten}', horizontalalignment='center', transform=axes[j, 1].transAxes, fontsize=35)
            #axes[j, 0].set_ylim(1e-44, 1e-38)
            axes[j, 0].set_ylim(1e-44, 1e-39)
        if target == 'iron':
            axes[j, 0].text(0.5, 0.94, r'\bf{Coherent Scattering off Iron}', horizontalalignment='center', transform=axes[j, 0].transAxes, fontsize=35)
            axes[j, 1].text(0.5, 0.94, r'\bf{Incoherent Scattering off Iron}', horizontalalignment='center', transform=axes[j, 1].transAxes, fontsize=35)
            #axes[j, 0].set_ylim(1e-44, 1e-39)
            axes[j, 0].set_ylim(1e-44, 1e-40)
    fig.legend(handles=trident_legend_lines, labels=all_labels, ncol=5, loc='outside upper center', fontsize=40)

fig.savefig('../plots/xsec.pdf',dpi=100)

if mue_tridents_flag:
    mue_colors = [color_numu_mu_e, color_nue_e_mu]
    mue_xsec = [numu_mu_e_dict, nue_e_mu_dict]
    mue_labels = [numu_mu_e_label, nue_e_mu_label]

    fig, ax = plt.subplots(1, 1, figsize=(15,12), tight_layout=True, sharex='col', sharey='row', squeeze=True)
    trident_legend_lines = [Line2D([0], [0], color=c, lw=5) for c in mue_colors]
    tungsten_legend_lines = [Line2D([0], [0], color='k', lw=5, linestyle=lnstyle) for lnstyle in ['solid','dashed']]
    iron_legend_lines = [Line2D([0], [0], color='k', lw=5, linestyle=lnstyle) for lnstyle in ['dotted','dashdot']]

    for i in range(2):
        xsec_dict = mue_xsec[i]
        trid_color = mue_colors[i]
        trid_label = mue_labels[i]
        for j, target in enumerate(SHiP_targets):
            energies = xsec_dict['coherent'][target][0]
            xsec_coh = xsec_dict['coherent'][target][1]
            energies_p, xsec_p = xsec_dict['proton']
            energies_n, xsec_n = xsec_dict['neutron']
            xsec_incoh = target_dict[target][1]*xsec_p + (target_dict[target][0]-target_dict[target][1])*xsec_n
            if target == 'tungsten':
                ax.plot(energies, np.divide(xsec_coh, energies), color = trid_color, label=trid_label, linestyle='solid', linewidth=6)
                ax.plot(energies_p, np.divide(xsec_incoh, energies_p), color = trid_color, label=trid_label, linestyle='dashed', linewidth=6)
            if target == 'iron':
                ax.plot(energies, np.divide(xsec_coh, energies), color = trid_color, label=trid_label, linestyle='dotted', linewidth=6)
                ax.plot(energies_p, np.divide(xsec_incoh, energies_p), color = trid_color, label=trid_label, linestyle='dashdot', linewidth=6)

            ax.set_xlim(1,400)
            ax.set_ylim(1e-44, 1e-39)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]', fontsize=50)
            ax.set_ylabel(r'Cross Section $\sigma/E_\nu$ [cm$^2$/GeV]', fontsize=50)
            ax.grid()

    tungsten_legend = ax.legend(handles=tungsten_legend_lines, labels=['Coherent','Incoherent'], ncols=1, loc='lower center', fontsize=30, title=r'\bf{Tungsten}')
    iron_legend = ax.legend(handles=iron_legend_lines, labels=['Coherent','Incoherent'], ncols=1, loc='lower right', fontsize=30, title=r'\bf{Iron}')
    ax.add_artist(tungsten_legend)
    ax.add_artist(iron_legend)
    ax.legend(handles=trident_legend_lines, labels=mue_labels, ncols=2, loc='upper center', fontsize=40)

    fig.savefig('../plots/xsec_mue.pdf',dpi=100)