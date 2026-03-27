import numpy as np

# helper functions
def interaction_events(flux: dict, cross_section: np.ndarray, area: float, detector_mass: float, target_mass: float) -> np.ndarray:
    """
    Calculate the number of interacting events given a cross section.
    \nInputs:
    - flux: 
        - if type is flux: number of incoming neutrinos at each bin [1]
        - if type is diff_flux: differential flux of incoming neutrinos per energy at each bin [GeV^-1]
    - cross_section: cross section of the process whose number of events will be calculated [cm^2]
    - area: detector cross sectional area [cm^2]
    - detector_mass: detector mass [kg]
    - target_mass: target mass [kg]
    \nOutput:
    - interaction_events [1]: number of interacting neutrinos at each bin for the given cross section process
    """
    probability = (cross_section/area) * (detector_mass/target_mass)
    if flux['type'] == 'flux':
        return flux['flux']*probability
    if flux['type'] == 'diff_flux':
        return flux['flux']*probability*np.diff(flux['bins'])


def total_event_count(event_spectrum: np.ndarray) -> float:
    """
    Calculate the total number of interaction events from a given event energy spectrum.
    \nInputs:
    - event_spectrum [1]: events energy spectrum
    \nOutput:
    - total [1]: total number of events sum(event_spectrum)
    """
    return np.sum(event_spectrum)

def interpolate(Elow: float, Ehigh: float, xsec_low: float, xsec_high: float, E: float) -> float:
    """
    Calculate the cross section for a given energy E given two reference energies and cross sections.
    Assume Elow <= E <= Ehigh
    \nInput:
    - Elow: first reference energy value
    - Ehigh: second reference energy value
    - xsec_low: first reference cross section
    - xsec_high: second reference cross section
    - E: energy at which cross section will be calculated
    \nOutput:
    - xsec: interpolated cross section given two energy values Elow and E2 with xsec1 and xsec2 respectively
    """
    assert Elow < Ehigh
    assert E >= Elow
    assert E <= Ehigh

    slope = (xsec_high - xsec_low)/(Ehigh-Elow)
    xsec = slope*(E-Elow) + xsec_low
    return xsec

def interpolate_xsec(energies:np.ndarray, ref_energies:np.ndarray, ref_xsec:np.ndarray) -> np.ndarray:
    """
    Interpolate the reference cross sections to the new energies and return the resulting interpolated cross section.
    \nInputs:
    - energies: array of new energies
    - ref_energies: array of reference energies
    - ref_xsec: array of cross sections evaluated at the reference energies
    \nOutput:
    - interp_xsec: array of cross sections evaluated at the new energies
    """
    size = len(energies)
    ref_size = len(ref_energies)
    interp_xsec = []

    for i in range(size):
        # If energies is larger than the maximum ref_energy, set interp_xsec to 0.
        if energies[i] > ref_energies[-1]: 
            interp_xsec.append(0.0)
            continue
        # If energy starts below ref_energy, set interp_xsec to 0. Otherwise, these values will be skipped.
        elif energies[i] < ref_energies[0]: 
            interp_xsec.append(0.0)
            continue
        else:
            for j in range(ref_size):
                # In the unlikely event that the new energy equals the reference energy, just output the cross section there.
                if energies[i] == ref_energies[j]: 
                    interp_xsec.append(ref_xsec[j])
                    continue
                # Find the reference energy 'bin' for the new energy and interpolate there.
                if ((energies[i] > ref_energies[j]) and (energies[i] < ref_energies[j+1])): 
                    # If ref_xsec is negative (it shouldn't be), just set interp_xsec to 0.
                    if (ref_xsec[j] < 0) or (ref_xsec[j+1] < 0): 
                        interp_xsec.append(0.0)
                    else:
                        interp = interpolate(ref_energies[j], ref_energies[j+1], ref_xsec[j], ref_xsec[j+1], energies[i])
                        interp_xsec.append(interp)
    
    assert len(energies) == len(interp_xsec)
    return np.array(interp_xsec)

class NEvents:
    def __init__(self, detector: str,
                 detector_mass: float,
                 target_mass: float,
                 detector_area: float,
                 exposure: float,
                 fluxes: dict | None = None,
                 xsec: dict | None = None,
                 events: dict | None = None) -> None:
        self.detector = detector
        self.detector_mass = detector_mass
        self.target_mass = target_mass
        self.detector_area = detector_area
        self.exposure = exposure
        if fluxes is None:
            self.fluxes = {}
        else:
            self.fluxes = fluxes
        if xsec is None:
            self.xsec = {}
        else:
            self.xsec = xsec
        if events is None:
            self.events = {}
        else:
            self.events = events
    
    def add_flux(self, flux_label: str, flux: np.ndarray, edges: np.ndarray, centers: np.ndarray, type: str = 'flux') -> None:
        f = {'flux': flux, 'bins': edges, 'centers': centers, 'type': type}
        self.fluxes[flux_label] = f

    def add_xsec(self, xsec_label: str, xsec: np.ndarray, energies: np.ndarray) -> None:
        x = {'xsec': xsec, 'energies': energies}
        self.xsec[xsec_label] = x

    def calculate_events(self, flux_label: str, xsec_label: str) -> None:
        # check whether the events for this cross section and this flux have been calculated before
        if flux_label+"+"+xsec_label not in self.events:
            # check whether the cross section has been interpolated to these flux energies before
            if xsec_label not in self.fluxes[flux_label]:
                # interpolated cross section to the flux energies
                interp_xsec = interpolate_xsec(self.fluxes[flux_label]['centers'],
                                               self.xsec[xsec_label]['energies'],
                                               self.xsec[xsec_label]['xsec'])
                self.fluxes[flux_label][xsec_label] = interp_xsec
            else:
                interp_xsec = self.fluxes[flux_label][xsec_label]
            # calculate event spectrum for this cross section using this flux with the detector's parameters
            event_spectrum = interaction_events(self.fluxes[flux_label],
                                                interp_xsec,
                                                self.detector_area,
                                                self.detector_mass,
                                                self.target_mass)
            # calculate the total number of events of this event spectrum
            total_events = total_event_count(event_spectrum)
            self.events[flux_label+"+"+xsec_label] = {'spectrum': event_spectrum, 'total': total_events}
    
    def print_total_events(self) -> None:
        print('Total events for ' + self.detector + ' (mass: ' + str(self.detector_mass) + ' kg, area: ' + str(self.detector_area) + ' cm^2, exposure: ' + str(self.exposure) + ' POT or fb^-1):\n')
        for process in self.events:
            print(process + ':  ' + str(self.events[process]['total']) + '\n')

    def save_total_events(self, filename: str) -> None:
        with open(filename, 'w') as file_object:
            print('Total events for ' + self.detector + ' (mass: ' + str(self.detector_mass) + ' kg, area: ' + str(self.detector_area) + ' cm^2, exposure: ' + str(self.exposure) + ' POT or fb^-1):\n', file=file_object)
            for process in self.events:
                total = self.events[process]['total']
                print(process + ':  ' + f'{total:.2e}' + '\n', file=file_object)