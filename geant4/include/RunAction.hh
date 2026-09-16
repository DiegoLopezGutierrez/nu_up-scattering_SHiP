#ifndef RunAction_h
#define RunAction_h 1

#include "G4UserRunAction.hh"

class G4Run;

// Books the HNL-kinematics ntuple and opens/closes the output ROOT file.
class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    ~RunAction() override = default;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;
};

#endif
