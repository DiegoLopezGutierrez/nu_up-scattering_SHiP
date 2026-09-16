#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"
#include "TrackingAction.hh"

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction());
  SetUserAction(new RunAction());
  SetUserAction(new SteppingAction());
  SetUserAction(new TrackingAction());
}
