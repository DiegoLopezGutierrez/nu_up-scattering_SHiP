#ifndef TrackingAction_h
#define TrackingAction_h 1

#include "G4UserTrackingAction.hh"
#include <memory>

class HNLProduction;

// ============================================================================
// TrackingAction
//
// Catches every charged pion / charged kaon track at the moment Geant4
// creates it (production point), regardless of which physics process or
// cascade generation produced it, and feeds its production 4-momentum to
// HNLProduction for the pi+/K+ -> l+ N two-body leptonic channels.
//
// Neutral kaons (K0_L, K0_S) and neutral pions have no tree-level
// charged-current two-body leptonic decay to l+N and are not processed.
// Charm/beauty mesons are handled separately in SteppingAction /
// Pythia8VertexModel, since Geant4 has no particle definition for them.
// ============================================================================

class TrackingAction : public G4UserTrackingAction
{
  public:
    TrackingAction();
    ~TrackingAction() override;

    void PreUserTrackingAction(const G4Track* track) override;

  private:
    std::unique_ptr<HNLProduction> fHNLProduction;
};

#endif
