from NEventsClass import NEvents
import csv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

plt.style.use('../plots/sty.mplstyle')

atomic_mass_unit = 1.6605e-27 # kg
tungsten_mass = 183.84*atomic_mass_unit # kg
iron_mass = 55.845*atomic_mass_unit # kg

coherent_flag = True
proton_flag = True
neutron_flag = True

SHiP_targets = ['tungsten', 'iron']
target_dict = {'tungsten':(184,74),
               'iron':(56,26),
               'argon': (40,18)}

# according to Antonio Iuliano's flux and to https://inspirehep.net/literature/3116831, the area is 40 cm x 40 cm
# according to that inspire paper, the length of the SND detector is 1.5 m.
# using these parameters, the detector's tungsten mass is 4.6 tonnes. The paper claims 3 tonnes.
# this is probably because the detector is made of alternating layers of silicone and tungsten so the total volume
# for tungsten would be less, leading to a smaller mass.
SHiP_mass = 3e3 # kg
SHiP_area = 40*40  # cm^2

### extract flux ###
SHiP_numu_numubar_flux = []
SHiP_nue_nuebar_flux = []
SHiP_nutau_nutaubar_flux = []
SHiP_edges = []
centers_SHiP = []

# the flux is normalized to 1 POT m^2 so we need to multiply by POT and detector area [m^2]
SHiP_POT = 6e20 # POT - equivalent to 15 years of running at 4e19 POT/year

with open('../flux/normalized_flux_at_detector.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',') 
    i = 0
    for row in data:
        if i == 0:
            i += 1
            continue
        SHiP_edges.append(float(row[0]))
        centers_SHiP.append((float(row[0]) + float(row[1]))/2.0)
        # there are 40 bins so save the right most bin edge at the last bin
        if i == 40:
            SHiP_edges.append(float(row[1]))
        # N neutrinos at each bin [1] normalized to 1 POT m^2 so we multiply by POT and detector area [m^2].
        SHiP_numu_numubar_flux.append((float(row[2]) + float(row[3])) * SHiP_POT * SHiP_area * 1e-4 )
        SHiP_nue_nuebar_flux.append((float(row[4]) + float(row[5])) * SHiP_POT * SHiP_area * 1e-4)
        SHiP_nutau_nutaubar_flux.append((float(row[6]) + float(row[7])) * SHiP_POT * SHiP_area * 1e-4)
        i += 1

SHiP_numu_numubar_flux = np.array(SHiP_numu_numubar_flux)
SHiP_nue_nuebar_flux = np.array(SHiP_nue_nuebar_flux)
SHiP_nutau_nutaubar_flux = np.array(SHiP_nutau_nutaubar_flux)
centers_SHiP = np.array(centers_SHiP)

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
    if proton:
        proton_filename = XSEC_DIR + process + '/nucleon/proton/' + process + '_nucleon_p_xsec.csv'
        energy_proton, xsec_proton = get_xsec(proton_filename)
        xsec_dict['proton'] = energy_proton, xsec_proton
    if neutron:
        neutron_filename = XSEC_DIR + process + '/nucleon/neutron/' + process + '_nucleon_n_xsec.csv'
        energy_neutron, xsec_neutron = get_xsec(neutron_filename)
        xsec_dict['neutron'] = energy_neutron, xsec_neutron
    for target in targets:
        xsec_dict[target] = {}
        if target == 'tungsten':
            tail = '_coh_W_xsec.csv'
        if target == 'argon':
            tail = '_coh_Ar_xsec.csv'
        if target == 'iron':
            tail = '_coh_Fe_xsec.csv'
        if coherent:
            coherent_filename = XSEC_DIR + process + '/coherent/' + target + '/' + process + tail
            xsec_dict[target]['coherent'] = get_xsec(coherent_filename)
        if proton and neutron:
            xsec_incoherent = target_dict[target][1]*xsec_proton + (target_dict[target][0]-target_dict[target][1])*xsec_neutron
            xsec_dict[target]['incoherent'] = energy_proton, xsec_incoherent
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


### get number of events ###
# SHiP
SHiP = NEvents('SHiP', SHiP_mass, tungsten_mass, SHiP_area, SHiP_POT)

SHiP.add_flux('numu_numubar', SHiP_numu_numubar_flux, SHiP_edges, centers_SHiP)
SHiP.add_flux('nue_nuebar', SHiP_nue_nuebar_flux, SHiP_edges, centers_SHiP)
SHiP.add_flux('nutau_nutaubar', SHiP_nutau_nutaubar_flux, SHiP_edges, centers_SHiP)

for target in SHiP_targets:
    # nue-initiated tridents
    SHiP.add_xsec(f'nue_e_e_coh_{target}', nue_e_e_dict[target]['coherent'][1], nue_e_e_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nue_e_e_incoh_{target}', nue_e_e_dict[target]['incoherent'][1], nue_e_e_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nue_mu_mu_coh_{target}', nue_mu_mu_dict[target]['coherent'][1], nue_mu_mu_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nue_mu_mu_incoh_{target}', nue_mu_mu_dict[target]['incoherent'][1], nue_mu_mu_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nue_tau_tau_coh_{target}', nue_tau_tau_dict[target]['coherent'][1], nue_tau_tau_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nue_tau_tau_incoh_{target}', nue_tau_tau_dict[target]['incoherent'][1], nue_tau_tau_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'numu_mu_e_coh_{target}', numu_mu_e_dict[target]['coherent'][1], numu_mu_e_dict[target]['coherent'][0])
    SHiP.add_xsec(f'numu_mu_e_incoh_{target}', numu_mu_e_dict[target]['incoherent'][1], numu_mu_e_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nutau_tau_e_coh_{target}', nutau_tau_e_dict[target]['coherent'][1], nutau_tau_e_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nutau_tau_e_incoh_{target}', nutau_tau_e_dict[target]['incoherent'][1], nutau_tau_e_dict[target]['incoherent'][0])

    # numu-initiated tridents
    SHiP.add_xsec(f'nue_e_mu_coh_{target}', nue_e_mu_dict[target]['coherent'][1], nue_e_mu_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nue_e_mu_incoh_{target}', nue_e_mu_dict[target]['incoherent'][1], nue_e_mu_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'numu_e_e_coh_{target}', numu_e_e_dict[target]['coherent'][1], numu_e_e_dict[target]['coherent'][0])
    SHiP.add_xsec(f'numu_e_e_incoh_{target}', numu_e_e_dict[target]['incoherent'][1], numu_e_e_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'numu_mu_mu_coh_{target}', numu_mu_mu_dict[target]['coherent'][1], numu_mu_mu_dict[target]['coherent'][0])
    SHiP.add_xsec(f'numu_mu_mu_incoh_{target}', numu_mu_mu_dict[target]['incoherent'][1], numu_mu_mu_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'numu_tau_tau_coh_{target}', numu_tau_tau_dict[target]['coherent'][1], numu_tau_tau_dict[target]['coherent'][0])
    SHiP.add_xsec(f'numu_tau_tau_incoh_{target}', numu_tau_tau_dict[target]['incoherent'][1], numu_tau_tau_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nutau_tau_mu_coh_{target}', nutau_tau_mu_dict[target]['coherent'][1], nutau_tau_mu_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nutau_tau_mu_incoh_{target}', nutau_tau_mu_dict[target]['incoherent'][1], nutau_tau_mu_dict[target]['incoherent'][0])

    # nutau-initiated tridents
    SHiP.add_xsec(f'nue_e_tau_coh_{target}', nue_e_tau_dict[target]['coherent'][1], nue_e_tau_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nue_e_tau_incoh_{target}', nue_e_tau_dict[target]['incoherent'][1], nue_e_tau_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'numu_mu_tau_coh_{target}', numu_mu_tau_dict[target]['coherent'][1], numu_mu_tau_dict[target]['coherent'][0])
    SHiP.add_xsec(f'numu_mu_tau_incoh_{target}', numu_mu_tau_dict[target]['incoherent'][1], numu_mu_tau_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nutau_e_e_coh_{target}', nutau_e_e_dict[target]['coherent'][1], nutau_e_e_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nutau_e_e_incoh_{target}', nutau_e_e_dict[target]['incoherent'][1], nutau_e_e_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nutau_mu_mu_coh_{target}', nutau_mu_mu_dict[target]['coherent'][1], nutau_mu_mu_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nutau_mu_mu_incoh_{target}', nutau_mu_mu_dict[target]['incoherent'][1], nutau_mu_mu_dict[target]['incoherent'][0])
    SHiP.add_xsec(f'nutau_tau_tau_coh_{target}', nutau_tau_tau_dict[target]['coherent'][1], nutau_tau_tau_dict[target]['coherent'][0])
    SHiP.add_xsec(f'nutau_tau_tau_incoh_{target}', nutau_tau_tau_dict[target]['incoherent'][1], nutau_tau_tau_dict[target]['incoherent'][0])

    # calculate events using flux
    SHiP.calculate_events('nue_nuebar',f'nue_e_e_coh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nue_e_e_incoh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nue_mu_mu_coh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nue_mu_mu_incoh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nue_tau_tau_coh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nue_tau_tau_incoh_{target}')
    SHiP.calculate_events('nue_nuebar',f'numu_mu_e_coh_{target}')
    SHiP.calculate_events('nue_nuebar',f'numu_mu_e_incoh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nutau_tau_e_coh_{target}')
    SHiP.calculate_events('nue_nuebar',f'nutau_tau_e_incoh_{target}')

    SHiP.calculate_events('numu_numubar',f'nue_e_mu_coh_{target}')
    SHiP.calculate_events('numu_numubar',f'nue_e_mu_incoh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_e_e_coh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_e_e_incoh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_mu_mu_coh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_mu_mu_incoh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_tau_tau_coh_{target}')
    SHiP.calculate_events('numu_numubar',f'numu_tau_tau_incoh_{target}')
    SHiP.calculate_events('numu_numubar',f'nutau_tau_mu_coh_{target}')
    SHiP.calculate_events('numu_numubar',f'nutau_tau_mu_incoh_{target}')

    SHiP.calculate_events('nutau_nutaubar',f'nue_e_tau_coh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nue_e_tau_incoh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'numu_mu_tau_coh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'numu_mu_tau_incoh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_e_e_coh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_e_e_incoh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_mu_mu_coh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_mu_mu_incoh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_tau_tau_coh_{target}')
    SHiP.calculate_events('nutau_nutaubar',f'nutau_tau_tau_incoh_{target}')

SHiP.save_total_events('SHiP_trident_events.txt')
SHiP.save_to_pandas('SHiP_trident_events_DataFrame.csv', index=False)

# plot event energy spectra
fig1, ax1 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True) # coherent tungsten
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True) # incoherent tungsten
fig3, ax3 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True) # coherent iron
fig4, ax4 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True) # incoherent iron

color_nue_e_e = '#4513D5'
color_nue_mu_mu    = '#DC3220'
color_nue_tau_tau = '#A8CE36'
color_numu_mu_e = '#A13FE5'
color_nutau_tau_e     = '#005AB5'

color_nue_e_mu     = '#299E14'
color_numu_e_e      = '#532E11'
color_numu_mu_mu  = '#EF2698'
color_numu_tau_tau    = '#FFA500'
color_nutau_tau_mu   = '#64B0FC'

color_nue_e_tau = '#482383'
color_numu_mu_tau = '#FB8E54'
color_nutau_e_e = '#C69C9D'
color_nutau_mu_mu = '#AD0E7D'
color_nutau_tau_tau = '#34CD66'

all_colors = [color_nue_e_e, color_nue_mu_mu, color_nue_tau_tau, color_numu_mu_e, color_nutau_tau_e,
              color_nue_e_mu, color_numu_e_e, color_numu_mu_mu, color_numu_tau_tau, color_nutau_tau_mu,
              color_nue_e_tau, color_numu_mu_tau,color_nutau_e_e, color_nutau_mu_mu, color_nutau_tau_tau]

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

all_labels = [nue_e_e_label, nue_mu_mu_label, nue_tau_tau_label, numu_mu_e_label, nutau_tau_e_label,
              nue_e_mu_label, numu_e_e_label, numu_mu_mu_label, numu_tau_tau_label, nutau_tau_mu_label,
              nue_e_tau_label, numu_mu_tau_label, nutau_e_e_label, nutau_mu_mu_label, nutau_tau_tau_label]

title_ypos = 1.0

trident_legend_lines = [Line2D([0], [0], color=c, lw=5) for c in all_colors]

# set plot parameters
for target in SHiP_targets:
    if target == 'tungsten':
        axes = [ax1, ax2]
        figs = [fig1, fig2]
        titles = [r"\bf{SHiP Coherent Trident Scattering - Tungsten}", r"\bf{SHiP Incoherent Trident Scattering - Tungsten}"]
    if target == 'iron':
        axes = [ax3, ax4]
        figs = [fig3, fig4]
        titles = [r"\bf{SHiP Coherent Trident Scattering - Iron}", r"\bf{SHiP Incoherent Trident Scattering - Iron}"]
    
    for i in range(2):
        axes[0].set_title(titles[0], fontsize=40) # add y = title_ypos if neq 1.0
        axes[1].set_title(titles[1], fontsize=40) # add y = title_ypos if neq 1.0

        axes[0].set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
        axes[0].set_ylabel(r"Neutrino Tridents $N_{\nu}^{\Psi}$")
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')

        axes[1].set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
        axes[1].set_ylabel(r"Neutrino Tridents $N_{\nu}^{\Psi}$")
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')

        axes[0].xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000))
        axes[0].xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000))
        axes[0].yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000))
        axes[0].yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000))

        axes[1].xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000))
        axes[1].xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000))
        axes[1].yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000))
        axes[1].yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000))

        axes[0].xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
        axes[0].yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

        axes[1].xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
        axes[1].yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

        axes[0].stairs(SHiP.events[f'nue_nuebar+nue_e_e_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_e_label, color=color_nue_e_e)
        axes[1].stairs(SHiP.events[f'nue_nuebar+nue_e_e_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_e_label, color=color_nue_e_e)
        axes[0].stairs(SHiP.events[f'nue_nuebar+nue_mu_mu_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_mu_mu_label, color=color_nue_mu_mu)
        axes[1].stairs(SHiP.events[f'nue_nuebar+nue_mu_mu_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_mu_mu_label, color=color_nue_mu_mu)
        axes[0].stairs(SHiP.events[f'nue_nuebar+nue_tau_tau_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_tau_tau_label, color=color_nue_tau_tau)
        axes[1].stairs(SHiP.events[f'nue_nuebar+nue_tau_tau_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_tau_tau_label, color=color_nue_tau_tau)
        axes[0].stairs(SHiP.events[f'nue_nuebar+numu_mu_e_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_e_label, color=color_numu_mu_e)
        axes[1].stairs(SHiP.events[f'nue_nuebar+numu_mu_e_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_e_label, color=color_numu_mu_e)
        axes[0].stairs(SHiP.events[f'nue_nuebar+nutau_tau_e_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_e_label, color=color_nutau_tau_e)
        axes[1].stairs(SHiP.events[f'nue_nuebar+nutau_tau_e_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_e_label, color=color_nutau_tau_e)

        axes[0].stairs(SHiP.events[f'numu_numubar+nue_e_mu_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_mu_label, color=color_nue_e_mu)
        axes[1].stairs(SHiP.events[f'numu_numubar+nue_e_mu_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_mu_label, color=color_nue_e_mu)
        axes[0].stairs(SHiP.events[f'numu_numubar+numu_e_e_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_e_e_label, color=color_numu_e_e)
        axes[1].stairs(SHiP.events[f'numu_numubar+numu_e_e_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_e_e_label, color=color_numu_e_e)
        axes[0].stairs(SHiP.events[f'numu_numubar+numu_mu_mu_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_mu_label, color=color_numu_mu_mu)
        axes[1].stairs(SHiP.events[f'numu_numubar+numu_mu_mu_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_mu_label, color=color_numu_mu_mu)
        axes[0].stairs(SHiP.events[f'numu_numubar+numu_tau_tau_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_tau_tau_label, color=color_numu_tau_tau)
        axes[1].stairs(SHiP.events[f'numu_numubar+numu_tau_tau_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_tau_tau_label, color=color_numu_tau_tau)
        axes[0].stairs(SHiP.events[f'numu_numubar+nutau_tau_mu_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_mu_label, color=color_nutau_tau_mu)
        axes[1].stairs(SHiP.events[f'numu_numubar+nutau_tau_mu_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_mu_label, color=color_nutau_tau_mu)

        axes[0].stairs(SHiP.events[f'nutau_nutaubar+nue_e_tau_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_tau_label, color=color_nue_e_tau)
        axes[1].stairs(SHiP.events[f'nutau_nutaubar+nue_e_tau_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_tau_label, color=color_nue_e_tau)
        axes[0].stairs(SHiP.events[f'nutau_nutaubar+numu_mu_tau_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_tau_label, color=color_numu_mu_tau)
        axes[1].stairs(SHiP.events[f'nutau_nutaubar+numu_mu_tau_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_tau_label, color=color_numu_mu_tau)
        axes[0].stairs(SHiP.events[f'nutau_nutaubar+nutau_e_e_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_e_e_label, color=color_nutau_e_e)
        axes[1].stairs(SHiP.events[f'nutau_nutaubar+nutau_e_e_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_e_e_label, color=color_nutau_e_e)
        axes[0].stairs(SHiP.events[f'nutau_nutaubar+nutau_mu_mu_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_mu_mu_label, color=color_nutau_mu_mu)
        axes[1].stairs(SHiP.events[f'nutau_nutaubar+nutau_mu_mu_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_mu_mu_label, color=color_nutau_mu_mu)
        axes[0].stairs(SHiP.events[f'nutau_nutaubar+nutau_tau_tau_coh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_tau_label, color=color_nutau_tau_tau)
        axes[1].stairs(SHiP.events[f'nutau_nutaubar+nutau_tau_tau_incoh_{target}']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_tau_label, color=color_nutau_tau_tau)

        axes[0].set_ylim(1e-6,1e3)
        axes[1].set_ylim(1e-6,1e3)

        axes[0].set_xlim(1, 400)
        axes[1].set_xlim(1, 400)

        axes[0].legend(handles=trident_legend_lines, labels=all_labels, loc='upper center', bbox_to_anchor=(0.5, title_ypos), ncol=5, frameon=False, fontsize=16.5)
        axes[1].legend(handles=trident_legend_lines, labels=all_labels, loc='upper center', bbox_to_anchor=(0.5, title_ypos), ncol=5, frameon=False, fontsize=16.5)

        figs[0].savefig(f"../plots/SHiP_coherent_events_{target}.pdf", dpi=100)
        figs[1].savefig(f"../plots/SHiP_incoherent_events_{target}.pdf", dpi=100)