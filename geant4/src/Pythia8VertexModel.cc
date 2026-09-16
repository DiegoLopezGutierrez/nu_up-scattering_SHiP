#include "Pythia8VertexModel.hh"
#include "HNLProduction.hh"

#include "Pythia8/Pythia.h"
#include "G4SystemOfUnits.hh"
#include "G4Exception.hh"
#include "Randomize.hh"

#include <array>
#include <cmath>
#include <sstream>

namespace {
  // Charged heavy-flavour mesons that Geant4 cannot represent as tracked
  // particles (no G4ParticleDefinition exists for them) but which are
  // exactly the mesons whose leptonic decays HNLProduction needs.
  const std::array<G4int, 4> kHeavyFlavorPDG = {411, 431, 521, 541}; // D+, Ds+, B+, Bc+

  // Representative lab energies [GeV] of the pool. The primary beam is
  // 400 GeV; lower bins cover secondary nucleons after earlier
  // generations of the cascade have degraded their energy. The floor is
  // set at 35 GeV lab (sqrt(s_NN) ~ 8.2 GeV): below this, ccbar
  // production is so close to threshold that Pythia8's phase-space
  // sampler becomes numerically unstable, and the true charm/beauty
  // yield is negligible anyway -- vertices below the floor are treated
  // as producing no charm/beauty at all (see ProcessVertex).

  // implementing finer linearly-spaced pool energies grid
  const std::array<G4double, 15> kPoolEnergiesGeV = 
      {400., 373., 347., 321., 295., 269., 243., 217., 191., 165., 139., 113., 87., 61., 35.};

  // const std::array<G4double, 6> kPoolEnergiesGeV =
  //     {400., 250., 150., 100., 60., 35.};

  // Fixed approximation for the total proton-proton inelastic cross
  // section [mb], used only as the normalization denominator for the
  // charm/beauty event weight (see Pythia8VertexModel.hh, note 2). Good
  // to ~20% across sqrt(s) ~ 8-30 GeV; refine with an energy-dependent
  // parameterization if better precision is needed.
  constexpr G4double kSigmaInelasticMb = 30.0;

  // Number of events generated at construction time to obtain a stable
  // Info::sigmaGen() estimate for each pool instance.
  constexpr int kBurnInEvents = 300;
}

Pythia8VertexModel::Pythia8VertexModel(HNLProduction* hnlProduction, G4int /*targetZ*/, G4int /*targetA*/)
  : fHNLProduction(hnlProduction)
{
  // targetZ/targetA are accepted for interface documentation purposes but
  // not used: every projectile/target nucleon is treated as a proton in
  // the pool below (isospin-symmetry approximation -- see header).
  for (const auto& eLab : kPoolEnergiesGeV) {
    fPoolEnergies.push_back(eLab);

    auto pythia = std::make_unique<Pythia8::Pythia>();

    // Fixed-energy, fixed-species proton-on-proton collision (isospin
    // symmetry stands in for a proper proton/neutron treatment -- see
    // Pythia8VertexModel.hh). Beam energy is fixed at construction time
    // because Beams:allowVariableEnergy is incompatible with the
    // explicit HardQCD processes below.
    pythia->readString("Beams:frameType = 3");
    pythia->readString("Beams:idA = 2212");
    pythia->readString("Beams:idB = 2212");
    pythia->settings.parm("Beams:pzA", eLab);
    pythia->readString("Beams:pzB = 0.");

    // Explicit hard-QCD charm/beauty pair production only -- this is the
    // piece missing from Geant4's own hadronic models. Deliberately *not*
    // combined with SoftQCD in the same instance (see header, note 2):
    // every generated event is guaranteed to contain a c-cbar or b-bbar
    // pair, and the physical per-vertex rate is recovered afterwards via
    // an explicit sigma(hard)/sigma(inelastic) weight.
    pythia->readString("HardQCD:hardccbar = on");
    pythia->readString("HardQCD:hardbbbar = on");

    // Let Pythia8 fully hadronize *and* decay everything with its own
    // decay tables. D/Ds/B/Bc do not need to survive as final-state
    // particles -- only their momentum where they appear in the event
    // record is needed (see ProcessVertex below), which is retained
    // whether or not they are later decayed.
    pythia->readString("HadronLevel:Decay = on");

    pythia->readString("Print:quiet = on");
    pythia->readString("Next:numberCount = 0");
    pythia->readString("Check:epTolErr = 0.1");

    // Cache the multiparton-interactions initialization (expensive to
    // recompute) across runs, one file per pool energy.
    std::ostringstream mpiFile;
    mpiFile << "pythia_target_mpi_" << static_cast<int>(eLab) << "GeV.dat";
    pythia->readString("MultipartonInteractions:reuseInit = 3");
    pythia->readString("MultipartonInteractions:initFile = " + mpiFile.str());

    if (!pythia->init()) {
      G4Exception("Pythia8VertexModel::Pythia8VertexModel", "Pythia8InitFailed",
                   FatalException, ("Pythia8 failed to initialize pool instance at "
                                     + std::to_string(eLab) + " GeV").c_str());
    }

    // Burn-in run to obtain a stable sigma(ccbar+bbbar) estimate; the
    // events themselves are discarded.
    for (int i = 0; i < kBurnInEvents; ++i) pythia->next();
    fSigmaHardMb.push_back(pythia->info.sigmaGen());

    fPool.push_back(std::move(pythia));
  }
}

Pythia8VertexModel::~Pythia8VertexModel() = default;

std::size_t Pythia8VertexModel::SelectInstanceIndex(G4double labEnergyGeV) const
{
  std::size_t best = 0;
  G4double bestDiff = std::abs(labEnergyGeV - fPoolEnergies[0]);
  for (std::size_t i = 1; i < fPoolEnergies.size(); ++i) {
    const G4double diff = std::abs(labEnergyGeV - fPoolEnergies[i]);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  }
  return best;
}

void Pythia8VertexModel::ProcessVertex(G4int /*projectilePDG*/,
                                        const G4LorentzVector& labMomentum,
                                        const G4ThreeVector& vertexPosition)
{
  const G4double labEnergyGeV = labMomentum.e() / CLHEP::GeV;

  // Below the lowest pool energy, treat the charm/beauty yield as zero
  // (see the kPoolEnergiesGeV comment above).
  if (labEnergyGeV < fPoolEnergies.back()) return;

  const std::size_t idx = SelectInstanceIndex(labEnergyGeV);
  Pythia8::Pythia& pythia = *fPool[idx];

  const G4double vertexWeight = fSigmaHardMb[idx] / kSigmaInelasticMb;
  if (vertexWeight <= 0.0) return;

  if (!pythia.next()) return;

  const G4ThreeVector labDirection = labMomentum.vect().unit();

  const Pythia8::Event& event = pythia.event;
  for (int i = 1; i < event.size(); ++i) {
    const int absId = std::abs(event[i].id());
    bool isHeavyFlavor = false;
    for (const auto& pdg : kHeavyFlavorPDG) {
      if (absId == pdg) { isHeavyFlavor = true; break; }
    }
    if (!isHeavyFlavor) continue;

    const Pythia8::Vec4 p = event[i].p();

    // Pythia8's pool instance has the beam along its own z-axis; rotate
    // into the true lab direction of the incident Geant4 track (a pure
    // rotation, so energy is unaffected -- see class-level comment).
    G4ThreeVector p3(p.px(), p.py(), p.pz());
    p3.rotateUz(labDirection);

    const G4LorentzVector labP4(p3 * CLHEP::GeV, p.e() * CLHEP::GeV);

    fHNLProduction->ProcessMeson(event[i].id(), labP4, vertexPosition, vertexWeight);
  }
}
