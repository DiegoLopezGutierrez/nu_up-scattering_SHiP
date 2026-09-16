#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <memory>

class HNLProduction;
class Pythia8VertexModel;

// ============================================================================
// SteppingAction
//
// Watches every step for a proton/neutron inelastic-scattering vertex
// (Geant4/FTFP_BERT continues to handle the real transport and produces
// the actual secondaries used for further tracking) and, at the vertex's
// incident energy, additionally calls Pythia8VertexModel purely to
// extract the charm/beauty content of that vertex for HNL-production
// bookkeeping (see Pythia8VertexModel.hh for why this is necessary).
// ============================================================================

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction();
    ~SteppingAction() override;

    void UserSteppingAction(const G4Step* step) override;

  private:
    std::unique_ptr<HNLProduction> fHNLProduction;
    std::unique_ptr<Pythia8VertexModel> fPythia8Model;
};

#endif
