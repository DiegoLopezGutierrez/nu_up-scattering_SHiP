#include "SteppingAction.hh"
#include "HNLProduction.hh"
#include "Pythia8VertexModel.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4LorentzVector.hh"

namespace {
  // Natural molybdenum: Z=42, <A>~96. Must match the target material in
  // DetectorConstruction. Treated as a free-nucleon gas for the purpose
  // of picking the struck nucleon in Pythia8VertexModel (see its header
  // for the corresponding approximation).
  constexpr G4int kTargetZ = 42;
  constexpr G4int kTargetA = 96;
}

SteppingAction::SteppingAction()
  : fHNLProduction(std::make_unique<HNLProduction>()),
    fPythia8Model(std::make_unique<Pythia8VertexModel>(fHNLProduction.get(), kTargetZ, kTargetA))
{}

SteppingAction::~SteppingAction() = default;

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  const G4int pdg = step->GetTrack()->GetDefinition()->GetPDGEncoding();
  if (pdg != 2212 && pdg != 2112) return; // only proton/neutron primaries of the cascade

  const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
  if (!proc) return;
  if (proc->GetProcessName().find("Inelastic") == std::string::npos) return;

  const G4StepPoint* preStep = step->GetPreStepPoint();
  const G4LorentzVector labMomentum(preStep->GetMomentum(), preStep->GetTotalEnergy());
  const G4ThreeVector vertexPosition = step->GetPostStepPoint()->GetPosition();

  fPythia8Model->ProcessVertex(pdg, labMomentum, vertexPosition);
}
