#ifndef Pythia8VertexModel_h
#define Pythia8VertexModel_h 1

#include "G4LorentzVector.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"
#include <vector>
#include <memory>

namespace Pythia8 { class Pythia; }
class HNLProduction;

// ============================================================================
// Pythia8VertexModel
//
// Geant4's stock hadronic models (FTFP_BERT and friends) do not produce
// charm or beauty hadrons, so charm/beauty HNL-production channels would
// silently be missing from a pure-Geant4 target simulation. This class
// plugs Pythia8 in *specifically* to supply the charm/beauty content of
// proton/neutron-nucleus inelastic vertices identified by SteppingAction,
// while Geant4/FTFP_BERT continues to handle the actual particle transport
// and the light-hadron (pi/K) cascade unmodified.
//
// Implementation notes
// ---------------------
// 1) Why a pool of fixed-energy instances: Pythia8's
//    Beams:allowVariableEnergy (used e.g. in Pythia8's own cosmic-ray
//    cascade example, examples/main183.cc, to cheaply re-use one Pythia
//    instance across many sub-collision energies) only supports SoftQCD
//    processes -- Pythia8 aborts initialization if it is combined with
//    explicit hard processes such as HardQCD:hardccbar/hardbbbar, which
//    are exactly what is needed here. Each proton/neutron generation in
//    the cascade loses energy, so a single fixed-energy instance is not
//    enough either. Instead this class keeps a small pool of fully-
//    initialized Pythia8 instances, one per representative lab energy;
//    at every vertex the nearest-energy instance is reused.
//
// 2) Why the hard process is generated as a *biased* sample, not
//    filtered from inclusive minimum bias: Pythia8 also refuses to
//    combine SoftQCD:all in the *same* instance as an explicit HardQCD
//    process ("should not combine softQCD processes with hard ones"),
//    and doing so anyway proved numerically unstable (intermittent
//    crashes in the phase-space sampler). Each pool instance therefore
//    runs *only* HardQCD:hardccbar/hardbbbar, i.e. every generated event
//    is guaranteed to contain a c-cbar or b-bbar pair. To turn this back
//    into a physical per-vertex rate, every heavy-flavour hit is weighted
//    by sigma(hard)/sigma(inelastic), where sigma(hard) is read off from
//    Pythia8 (Info::sigmaGen(), averaged over a short burn-in run at
//    construction time) and sigma(inelastic) is a fixed, energy-
//    independent approximation (see kSigmaInelasticMb in the .cc) --
//    accurate to ~20% over the SPS energy range, which is adequate for a
//    phenomenology-level flux estimate. This biased-sampling approach is
//    also far more statistically efficient than waiting for charm/beauty
//    to spontaneously appear in inclusive minimum-bias events, given how
//    rare true charm/beauty production is per vertex.
//
// 3) Two further simplifications, both driven by the same constraint
//    (fixed beam species per pool instance, since Beams:allowIDAswitch
//    also requires allowVariableEnergy):
//     - Every projectile/target nucleon is treated as a proton for this
//       charm/beauty sub-calculation (isospin symmetry: proton-nucleon
//       and neutron-nucleon differ only at the current-quark level via
//       the parton distributions, a small effect relative to the other
//       approximations already made here).
//     - The target nucleus is treated as a free-nucleon gas at rest (no
//       nuclear shadowing / Fermi motion / Glauber multi-nucleon
//       treatment).
//    Since Pythia8's internal frame for each pool instance has the beam
//    along its own z-axis, the produced 4-momenta are rotated into the
//    *actual* lab direction of the incident Geant4 track before being
//    handed to HNLProduction.
// ============================================================================

class Pythia8VertexModel
{
  public:
    Pythia8VertexModel(HNLProduction* hnlProduction, G4int targetZ, G4int targetA);
    ~Pythia8VertexModel();

    // Generate one hadron-nucleon sub-collision for the given projectile
    // (proton or neutron) at its current lab 4-momentum, and forward any
    // produced D+/Ds+/B+/Bc+ to HNLProduction. vertexPosition is only
    // carried through for bookkeeping in the output ntuple.
    void ProcessVertex(G4int projectilePDG,
                        const G4LorentzVector& labMomentum,
                        const G4ThreeVector& vertexPosition);

  private:
    // Representative lab energies [GeV] of the pool, ascending.
    std::vector<G4double> fPoolEnergies;
    std::vector<std::unique_ptr<Pythia8::Pythia>> fPool;

    // sigma(ccbar+bbbar) [mb] for each pool instance, from Info::sigmaGen()
    // after a short burn-in run at construction time.
    std::vector<G4double> fSigmaHardMb;

    HNLProduction* fHNLProduction;

    std::size_t SelectInstanceIndex(G4double labEnergyGeV) const;
};

#endif
