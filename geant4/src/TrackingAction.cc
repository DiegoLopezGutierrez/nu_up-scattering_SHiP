#include "TrackingAction.hh"
#include "HNLProduction.hh"

#include "G4Track.hh"
#include "G4LorentzVector.hh"

TrackingAction::TrackingAction()
  : fHNLProduction(std::make_unique<HNLProduction>())
{}

TrackingAction::~TrackingAction() = default;

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
  const G4int pdg = track->GetDefinition()->GetPDGEncoding();
  if (pdg != 211 && pdg != -211 && pdg != 321 && pdg != -321) return;

  const G4LorentzVector labMomentum(track->GetMomentum(), track->GetTotalEnergy());
  fHNLProduction->ProcessMeson(pdg, labMomentum, track->GetPosition());
}
