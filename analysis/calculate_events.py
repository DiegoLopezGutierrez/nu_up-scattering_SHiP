from NEventsClass import NEvents
import csv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('../plots/sty.mplstyle')

atomic_mass_unit = 1.6605e-27 # kg
tungsten_mass = 183.84*atomic_mass_unit # kg
coherent_flag = True
proton_flag = True
neutron_flag = True

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
SHiP_numu_numubar_diff_flux = []
SHiP_nue_nuebar_diff_flux = []
SHiP_nutau_nutaubar_diff_flux = []
SHiP_edges = []
centers_SHiP = []

# the flux is normalized to 1 POT m^2 so we need to multiply by POT and detector area [m^2]
SHiP_POT = 2e20

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

with open('../flux/normalized_diff_flux_at_detector.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',') 
    i = 0
    for row in data:
        if i == 0:
            i += 1
            continue
        # dN/dE neutrinos at each bin [GeV^-1] normalized to 1 POT m^2 so we multiply by POT and detector area [m^2].
        SHiP_numu_numubar_diff_flux.append((float(row[2]) + float(row[3])) * SHiP_POT * SHiP_area * 1e-4 )
        SHiP_nue_nuebar_diff_flux.append((float(row[4]) + float(row[5])) * SHiP_POT * SHiP_area * 1e-4)
        SHiP_nutau_nutaubar_diff_flux.append((float(row[6]) + float(row[7])) * SHiP_POT * SHiP_area * 1e-4)
        i += 1

SHiP_numu_numubar_diff_flux = np.array(SHiP_numu_numubar_diff_flux)
SHiP_nue_nuebar_diff_flux = np.array(SHiP_nue_nuebar_diff_flux)
SHiP_nutau_nutaubar_diff_flux = np.array(SHiP_nutau_nutaubar_diff_flux)

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

def get_xsecs(process: str, coherent: bool = True, proton: bool = True, neutron: bool = True, target: str = 'tungsten') -> dict:
    XSEC_DIR = '../cross_sections/'
    xsec_dict = {}
    if target == 'tungsten':
        A = 184
        Z = 74
        tail = '_coh_W_xsec.csv'
    if target == 'argon':
        A = 40
        Z = 18
        tail = '_coh_Ar_xsec.csv'
    if coherent:
        coherent_filename = XSEC_DIR + process + '/coherent/' + target + '/' + process + tail
        coh_energy, coh_xsec = get_xsec(coherent_filename)
        xsec_dict['coherent'] = coh_energy, coh_xsec
    if proton:
        proton_filename = XSEC_DIR + process + '/nucleon/proton/' + process + '_nucleon_p_xsec.csv'
        p_energy, p_xsec = get_xsec(proton_filename, factor=Z)
        xsec_dict['proton'] = p_energy, p_xsec
    if neutron:
        neutron_filename = XSEC_DIR + process + '/nucleon/neutron/' + process + '_nucleon_n_xsec.csv'
        n_energy, n_xsec = get_xsec(neutron_filename, factor=(A-Z))
        xsec_dict['neutron'] = n_energy, n_xsec
    if proton and neutron:
        xsec_dict['incoherent'] = p_energy, (p_xsec + n_xsec)
    return xsec_dict

# nue-initiated tridents
nue_e_e_dict = get_xsecs('ve_to_ve_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nue_mu_mu_dict = get_xsecs('ve_to_ve_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nue_tau_tau_dict = get_xsecs('ve_to_ve_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
numu_mu_e_dict = get_xsecs('ve_to_vmu_mu+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nutau_tau_e_dict = get_xsecs('ve_to_vtau_tau+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')

# numu-initiated tridents
nue_e_mu_dict = get_xsecs('vmu_to_ve_e+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
numu_e_e_dict = get_xsecs('vmu_to_vmu_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
numu_mu_mu_dict = get_xsecs('vmu_to_vmu_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
numu_tau_tau_dict = get_xsecs('vmu_to_vmu_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nutau_tau_mu_dict = get_xsecs('vmu_to_vtau_tau+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')

# nutau-initiated tridents
nue_e_tau_dict = get_xsecs('vtau_to_ve_e+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
numu_mu_tau_dict = get_xsecs('vtau_to_vmu_mu+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nutau_e_e_dict = get_xsecs('vtau_to_vtau_e+_e-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nutau_mu_mu_dict = get_xsecs('vtau_to_vtau_mu+_mu-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')
nutau_tau_tau_dict = get_xsecs('vtau_to_vtau_tau+_tau-', coherent=coherent_flag, proton=proton_flag, neutron=neutron_flag, target='tungsten')

### get number of events ###
# SHiP
SHiP = NEvents('SHiP', SHiP_mass, tungsten_mass, SHiP_area, SHiP_POT)

SHiP.add_flux('numu_numubar', SHiP_numu_numubar_flux, SHiP_edges, centers_SHiP)
SHiP.add_flux('nue_nuebar', SHiP_nue_nuebar_flux, SHiP_edges, centers_SHiP)
SHiP.add_flux('nutau_nutaubar', SHiP_nutau_nutaubar_flux, SHiP_edges, centers_SHiP)
SHiP.add_flux('diff_numu_numubar', SHiP_numu_numubar_diff_flux, SHiP_edges, centers_SHiP, type='diff_flux')
SHiP.add_flux('diff_nue_nuebar', SHiP_nue_nuebar_diff_flux, SHiP_edges, centers_SHiP, type='diff_flux')
SHiP.add_flux('diff_nutau_nutaubar', SHiP_nutau_nutaubar_diff_flux, SHiP_edges, centers_SHiP, type='diff_flux')

# nue-initiated tridents
SHiP.add_xsec('nue_e_e_coh', nue_e_e_dict['coherent'][1], nue_e_e_dict['coherent'][0])
SHiP.add_xsec('nue_e_e_incoh', nue_e_e_dict['incoherent'][1], nue_e_e_dict['incoherent'][0])
SHiP.add_xsec('nue_mu_mu_coh', nue_mu_mu_dict['coherent'][1], nue_mu_mu_dict['coherent'][0])
SHiP.add_xsec('nue_mu_mu_incoh', nue_mu_mu_dict['incoherent'][1], nue_mu_mu_dict['incoherent'][0])
SHiP.add_xsec('nue_tau_tau_coh', nue_tau_tau_dict['coherent'][1], nue_tau_tau_dict['coherent'][0])
SHiP.add_xsec('nue_tau_tau_incoh', nue_tau_tau_dict['incoherent'][1], nue_tau_tau_dict['incoherent'][0])
SHiP.add_xsec('numu_mu_e_coh', numu_mu_e_dict['coherent'][1], numu_mu_e_dict['coherent'][0])
SHiP.add_xsec('numu_mu_e_incoh', numu_mu_e_dict['incoherent'][1], numu_mu_e_dict['incoherent'][0])
SHiP.add_xsec('nutau_tau_e_coh', nutau_tau_e_dict['coherent'][1], nutau_tau_e_dict['coherent'][0])
SHiP.add_xsec('nutau_tau_e_incoh', nutau_tau_e_dict['incoherent'][1], nutau_tau_e_dict['incoherent'][0])

# numu-initiated tridents
SHiP.add_xsec('nue_e_mu_coh', nue_e_mu_dict['coherent'][1], nue_e_mu_dict['coherent'][0])
SHiP.add_xsec('nue_e_mu_incoh', nue_e_mu_dict['incoherent'][1], nue_e_mu_dict['incoherent'][0])
SHiP.add_xsec('numu_e_e_coh', numu_e_e_dict['coherent'][1], numu_e_e_dict['coherent'][0])
SHiP.add_xsec('numu_e_e_incoh', numu_e_e_dict['incoherent'][1], numu_e_e_dict['incoherent'][0])
SHiP.add_xsec('numu_mu_mu_coh', numu_mu_mu_dict['coherent'][1], numu_mu_mu_dict['coherent'][0])
SHiP.add_xsec('numu_mu_mu_incoh', numu_mu_mu_dict['incoherent'][1], numu_mu_mu_dict['incoherent'][0])
SHiP.add_xsec('numu_tau_tau_coh', numu_tau_tau_dict['coherent'][1], numu_tau_tau_dict['coherent'][0])
SHiP.add_xsec('numu_tau_tau_incoh', numu_tau_tau_dict['incoherent'][1], numu_tau_tau_dict['incoherent'][0])
SHiP.add_xsec('nutau_tau_mu_coh', nutau_tau_mu_dict['coherent'][1], nutau_tau_mu_dict['coherent'][0])
SHiP.add_xsec('nutau_tau_mu_incoh', nutau_tau_mu_dict['incoherent'][1], nutau_tau_mu_dict['incoherent'][0])

# nutau-initiated tridents
SHiP.add_xsec('nue_e_tau_coh', nue_e_tau_dict['coherent'][1], nue_e_tau_dict['coherent'][0])
SHiP.add_xsec('nue_e_tau_incoh', nue_e_tau_dict['incoherent'][1], nue_e_tau_dict['incoherent'][0])
SHiP.add_xsec('numu_mu_tau_coh', numu_mu_tau_dict['coherent'][1], numu_mu_tau_dict['coherent'][0])
SHiP.add_xsec('numu_mu_tau_incoh', numu_mu_tau_dict['incoherent'][1], numu_mu_tau_dict['incoherent'][0])
SHiP.add_xsec('nutau_e_e_coh', nutau_e_e_dict['coherent'][1], nutau_e_e_dict['coherent'][0])
SHiP.add_xsec('nutau_e_e_incoh', nutau_e_e_dict['incoherent'][1], nutau_e_e_dict['incoherent'][0])
SHiP.add_xsec('nutau_mu_mu_coh', nutau_mu_mu_dict['coherent'][1], nutau_mu_mu_dict['coherent'][0])
SHiP.add_xsec('nutau_mu_mu_incoh', nutau_mu_mu_dict['incoherent'][1], nutau_mu_mu_dict['incoherent'][0])
SHiP.add_xsec('nutau_tau_tau_coh', nutau_tau_tau_dict['coherent'][1], nutau_tau_tau_dict['coherent'][0])
SHiP.add_xsec('nutau_tau_tau_incoh', nutau_tau_tau_dict['incoherent'][1], nutau_tau_tau_dict['incoherent'][0])


# calculate events using flux
SHiP.calculate_events('nue_nuebar','nue_e_e_coh')
SHiP.calculate_events('nue_nuebar','nue_e_e_incoh')
SHiP.calculate_events('nue_nuebar','nue_mu_mu_coh')
SHiP.calculate_events('nue_nuebar','nue_mu_mu_incoh')
SHiP.calculate_events('nue_nuebar','nue_tau_tau_coh')
SHiP.calculate_events('nue_nuebar','nue_tau_tau_incoh')
SHiP.calculate_events('nue_nuebar','numu_mu_e_coh')
SHiP.calculate_events('nue_nuebar','numu_mu_e_incoh')
SHiP.calculate_events('nue_nuebar','nutau_tau_e_coh')
SHiP.calculate_events('nue_nuebar','nutau_tau_e_incoh')

SHiP.calculate_events('numu_numubar','nue_e_mu_coh')
SHiP.calculate_events('numu_numubar','nue_e_mu_incoh')
SHiP.calculate_events('numu_numubar','numu_e_e_coh')
SHiP.calculate_events('numu_numubar','numu_e_e_incoh')
SHiP.calculate_events('numu_numubar','numu_mu_mu_coh')
SHiP.calculate_events('numu_numubar','numu_mu_mu_incoh')
SHiP.calculate_events('numu_numubar','numu_tau_tau_coh')
SHiP.calculate_events('numu_numubar','numu_tau_tau_incoh')
SHiP.calculate_events('numu_numubar','nutau_tau_mu_coh')
SHiP.calculate_events('numu_numubar','nutau_tau_mu_incoh')

SHiP.calculate_events('nutau_nutaubar','nue_e_tau_coh')
SHiP.calculate_events('nutau_nutaubar','nue_e_tau_incoh')
SHiP.calculate_events('nutau_nutaubar','numu_mu_tau_coh')
SHiP.calculate_events('nutau_nutaubar','numu_mu_tau_incoh')
SHiP.calculate_events('nutau_nutaubar','nutau_e_e_coh')
SHiP.calculate_events('nutau_nutaubar','nutau_e_e_incoh')
SHiP.calculate_events('nutau_nutaubar','nutau_mu_mu_coh')
SHiP.calculate_events('nutau_nutaubar','nutau_mu_mu_incoh')
SHiP.calculate_events('nutau_nutaubar','nutau_tau_tau_coh')
SHiP.calculate_events('nutau_nutaubar','nutau_tau_tau_incoh')

# calculate events using differential flux
SHiP.calculate_events('diff_nue_nuebar','nue_e_e_coh')
SHiP.calculate_events('diff_nue_nuebar','nue_e_e_incoh')
SHiP.calculate_events('diff_nue_nuebar','nue_mu_mu_coh')
SHiP.calculate_events('diff_nue_nuebar','nue_mu_mu_incoh')
SHiP.calculate_events('diff_nue_nuebar','nue_tau_tau_coh')
SHiP.calculate_events('diff_nue_nuebar','nue_tau_tau_incoh')
SHiP.calculate_events('diff_nue_nuebar','numu_mu_e_coh')
SHiP.calculate_events('diff_nue_nuebar','numu_mu_e_incoh')
SHiP.calculate_events('diff_nue_nuebar','nutau_tau_e_coh')
SHiP.calculate_events('diff_nue_nuebar','nutau_tau_e_incoh')

SHiP.calculate_events('diff_numu_numubar','nue_e_mu_coh')
SHiP.calculate_events('diff_numu_numubar','nue_e_mu_incoh')
SHiP.calculate_events('diff_numu_numubar','numu_e_e_coh')
SHiP.calculate_events('diff_numu_numubar','numu_e_e_incoh')
SHiP.calculate_events('diff_numu_numubar','numu_mu_mu_coh')
SHiP.calculate_events('diff_numu_numubar','numu_mu_mu_incoh')
SHiP.calculate_events('diff_numu_numubar','numu_tau_tau_coh')
SHiP.calculate_events('diff_numu_numubar','numu_tau_tau_incoh')
SHiP.calculate_events('diff_numu_numubar','nutau_tau_mu_coh')
SHiP.calculate_events('diff_numu_numubar','nutau_tau_mu_incoh')

SHiP.calculate_events('diff_nutau_nutaubar','nue_e_tau_coh')
SHiP.calculate_events('diff_nutau_nutaubar','nue_e_tau_incoh')
SHiP.calculate_events('diff_nutau_nutaubar','numu_mu_tau_coh')
SHiP.calculate_events('diff_nutau_nutaubar','numu_mu_tau_incoh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_e_e_coh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_e_e_incoh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_mu_mu_coh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_mu_mu_incoh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_tau_tau_coh')
SHiP.calculate_events('diff_nutau_nutaubar','nutau_tau_tau_incoh')

SHiP.save_total_events('SHiP_trident_events.txt')

# plot event energy spectra
fig1, ax1 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True)
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12), tight_layout=True)

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

title_ypos = 1.0
ax1.set_title(r"\bf{SHiP Coherent Trident Scattering}", fontsize=45) # add y = title_ypos if neq 1.0
ax2.set_title(r"\bf{SHiP Incoherent Trident Scattering}", fontsize=45) # add y = title_ypos if neq 1.0

ax1.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
ax1.set_ylabel(r"Neutrino Tridents $N_{\nu}^{\Psi}$")
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
ax2.set_ylabel(r"Neutrino Tridents $N_{\nu}^{\Psi}$")
ax2.set_xscale('log')
ax2.set_yscale('log')

locmaj_x1 = mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000)
locmin_x1 = mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000)
locmaj_y1 = mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000)
locmin_y1 = mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000)

locmaj_x2 = mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000)
locmin_x2 = mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000)
locmaj_y2 = mticker.LogLocator(base=10.0, subs=(1.0, ), numticks=1000)
locmin_y2 = mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=2000)

ax1.xaxis.set_major_locator(locmaj_x1)
ax1.xaxis.set_minor_locator(locmin_x1)
ax1.yaxis.set_major_locator(locmaj_y1)
ax1.yaxis.set_minor_locator(locmin_y1)

ax2.xaxis.set_major_locator(locmaj_x2)
ax2.xaxis.set_minor_locator(locmin_x2)
ax2.yaxis.set_major_locator(locmaj_y2)
ax2.yaxis.set_minor_locator(locmin_y2)

ax1.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax1.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax2.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)
ax2.yaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.45)

ax1.stairs(SHiP.events['nue_nuebar+nue_e_e_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_e_label, color=color_nue_e_e)
ax2.stairs(SHiP.events['nue_nuebar+nue_e_e_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_e_label, color=color_nue_e_e)
ax1.stairs(SHiP.events['nue_nuebar+nue_mu_mu_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_mu_mu_label, color=color_nue_mu_mu)
ax2.stairs(SHiP.events['nue_nuebar+nue_mu_mu_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_mu_mu_label, color=color_nue_mu_mu)
ax1.stairs(SHiP.events['nue_nuebar+nue_tau_tau_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_tau_tau_label, color=color_nue_tau_tau)
ax2.stairs(SHiP.events['nue_nuebar+nue_tau_tau_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_tau_tau_label, color=color_nue_tau_tau)
ax1.stairs(SHiP.events['nue_nuebar+numu_mu_e_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_e_label, color=color_numu_mu_e)
ax2.stairs(SHiP.events['nue_nuebar+numu_mu_e_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_e_label, color=color_numu_mu_e)
ax1.stairs(SHiP.events['nue_nuebar+nutau_tau_e_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_e_label, color=color_nutau_tau_e)
ax2.stairs(SHiP.events['nue_nuebar+nutau_tau_e_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_e_label, color=color_nutau_tau_e)

ax1.stairs(SHiP.events['numu_numubar+nue_e_mu_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_mu_label, color=color_nue_e_mu)
ax2.stairs(SHiP.events['numu_numubar+nue_e_mu_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_mu_label, color=color_nue_e_mu)
ax1.stairs(SHiP.events['numu_numubar+numu_e_e_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_e_e_label, color=color_numu_e_e)
ax2.stairs(SHiP.events['numu_numubar+numu_e_e_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_e_e_label, color=color_numu_e_e)
ax1.stairs(SHiP.events['numu_numubar+numu_mu_mu_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_mu_label, color=color_numu_mu_mu)
ax2.stairs(SHiP.events['numu_numubar+numu_mu_mu_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_mu_label, color=color_numu_mu_mu)
ax1.stairs(SHiP.events['numu_numubar+numu_tau_tau_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_tau_tau_label, color=color_numu_tau_tau)
ax2.stairs(SHiP.events['numu_numubar+numu_tau_tau_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_tau_tau_label, color=color_numu_tau_tau)
ax1.stairs(SHiP.events['numu_numubar+nutau_tau_mu_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_mu_label, color=color_nutau_tau_mu)
ax2.stairs(SHiP.events['numu_numubar+nutau_tau_mu_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_mu_label, color=color_nutau_tau_mu)

ax1.stairs(SHiP.events['nutau_nutaubar+nue_e_tau_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_tau_label, color=color_nue_e_tau)
ax2.stairs(SHiP.events['nutau_nutaubar+nue_e_tau_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nue_e_tau_label, color=color_nue_e_tau)
ax1.stairs(SHiP.events['nutau_nutaubar+numu_mu_tau_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_tau_label, color=color_numu_mu_tau)
ax2.stairs(SHiP.events['nutau_nutaubar+numu_mu_tau_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=numu_mu_tau_label, color=color_numu_mu_tau)
ax1.stairs(SHiP.events['nutau_nutaubar+nutau_e_e_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_e_e_label, color=color_nutau_e_e)
ax2.stairs(SHiP.events['nutau_nutaubar+nutau_e_e_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_e_e_label, color=color_nutau_e_e)
ax1.stairs(SHiP.events['nutau_nutaubar+nutau_mu_mu_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_mu_mu_label, color=color_nutau_mu_mu)
ax2.stairs(SHiP.events['nutau_nutaubar+nutau_mu_mu_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_mu_mu_label, color=color_nutau_mu_mu)
ax1.stairs(SHiP.events['nutau_nutaubar+nutau_tau_tau_coh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_tau_label, color=color_nutau_tau_tau)
ax2.stairs(SHiP.events['nutau_nutaubar+nutau_tau_tau_incoh']['spectrum'], SHiP_edges, lw=5, alpha=1, label=nutau_tau_tau_label, color=color_nutau_tau_tau)

ax1.set_ylim(1e-6,1e3)
ax2.set_ylim(1e-6,1e3)

ax1.set_xlim(1, 400)
ax2.set_xlim(1, 400)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, title_ypos), ncol=5, frameon=False, fontsize=16.5)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, title_ypos), ncol=5, frameon=False, fontsize=16.5)

fig1.savefig("../plots/SHiP_coherent_events.pdf", dpi=100)
fig2.savefig("../plots/SHiP_incoherent_events.pdf", dpi=100)