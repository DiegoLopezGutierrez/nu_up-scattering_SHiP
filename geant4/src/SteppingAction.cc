#include "SteppingAction.hh"
#include "HNLProduction.hh"
#include "Pythia8VertexModel.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4LorentzVector.hh"
#include "G4ios.hh"

namespace {
  // Natural molybdenum: Z=42, <A>~96. Must match the target material in
  // DetectorConstruction. Treated as a free-nucleon gas for the purpose
  // of picking the struck nucleon in Pythia8VertexModel (see its header
  // for the corresponding approximation).
  constexpr G4int kTargetZ = 42;
  constexpr G4int kTargetA = 96;

  // --- Diagnostic instrumentation (commented out, kept for future use):
  // how many inelastic cascade vertices per primary proton actually clear
  // the 35 GeV charm-production floor (Pythia8VertexModel::ProcessVertex
  // silently drops anything below it), and the nucleon/meson species split
  // of those, to check against reference [64]'s quoted split (65%/35%
  // charm, 77%/23% beauty). Used to track down the production-rate gap
  // flagged against arXiv:1811.00930's Table 1+2 inputs -- see
  // ../PRODUCTION_RATE_GAP.md. Uncomment to re-enable.
  // constexpr G4double kPoolFloorGeV = 35.0;
  // G4long gTotalVertices = 0;
  // G4long gVerticesAboveFloor = 0;
  // G4double gMaxEnergySeen = 0.0;
  // G4long gAboveFloorNucleon = 0;
  // G4long gAboveFloorMeson = 0;
  //
  // bool IsNucleon(G4int pdg)
  // {
  //   const G4int absPdg = std::abs(pdg);
  //   return absPdg == 2212 || absPdg == 2112;
  // }

  // Cascade beam species forwarded to Pythia8VertexModel: proton, neutron,
  // pi+/-, K+/- (see ../PRODUCTION_RATE_GAP.md, Finding 1 -- the true
  // cascade also includes these mesons, not just nucleons). KS/KL are not
  // included, a documented scope limitation.
  bool IsCascadeBeamSpecies(G4int pdg)
  {
    const G4int absPdg = std::abs(pdg);
    return absPdg == 2212 || absPdg == 2112 || absPdg == 211 || absPdg == 321;
  }
}

SteppingAction::SteppingAction()
  : fHNLProduction(std::make_unique<HNLProduction>()),
    fPythia8Model(std::make_unique<Pythia8VertexModel>(fHNLProduction.get(), kTargetZ, kTargetA))
{}

SteppingAction::~SteppingAction()
{
  // --- Diagnostic instrumentation (commented out, kept for future use):
  // G4cout << "\n=== SteppingAction vertex-floor diagnostic ===\n"
  //        << "Total cascade (p/n/pi+-/K+-) inelastic vertices: " << gTotalVertices << "\n"
  //        << "Vertices with E_lab >= " << kPoolFloorGeV << " GeV: " << gVerticesAboveFloor << "\n"
  //        << "  of which nucleon-initiated (p/n): " << gAboveFloorNucleon << "\n"
  //        << "  of which meson-initiated (pi/K): " << gAboveFloorMeson << "\n"
  //        << "Max single-vertex energy seen [GeV]: " << gMaxEnergySeen << "\n"
  //        << "===============================================\n" << G4endl;
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  const G4int pdg = step->GetTrack()->GetDefinition()->GetPDGEncoding();
  if (!IsCascadeBeamSpecies(pdg)) return; // p, n, pi+/-, K+/- cascade particles only

  const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
  if (!proc) return;
  if (proc->GetProcessName().find("Inelastic") == std::string::npos) return;

  const G4StepPoint* preStep = step->GetPreStepPoint();
  const G4LorentzVector labMomentum(preStep->GetMomentum(), preStep->GetTotalEnergy());
  const G4ThreeVector vertexPosition = step->GetPostStepPoint()->GetPosition();

  // --- diagnostic bookkeeping (commented out, kept for future use) ---
  // const G4double eLabGeV = labMomentum.e() / CLHEP::GeV;
  // gTotalVertices++;
  // if (eLabGeV > gMaxEnergySeen) gMaxEnergySeen = eLabGeV;
  // if (eLabGeV >= kPoolFloorGeV) {
  //   gVerticesAboveFloor++;
  //   if (IsNucleon(pdg)) gAboveFloorNucleon++;
  //   else gAboveFloorMeson++;
  // }
  // --- end diagnostic bookkeeping ---

  fPythia8Model->ProcessVertex(pdg, labMomentum, vertexPosition);
}
