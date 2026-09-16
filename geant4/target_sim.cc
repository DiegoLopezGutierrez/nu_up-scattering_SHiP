// HNL production-flux estimate for the SHiP proton target.
//
// Usage:
//   ./target_sim [nProtons]
//
// Simulates nProtons (default 1000) 400 GeV protons hitting a cylindrical
// molybdenum target (r = 125 mm, length = 60 cm) with Geant4/FTFP_BERT,
// supplementing Geant4's hadronic models with Pythia8 for charm/beauty
// production (see Pythia8VertexModel.hh), and writes HNL-production
// kinematics implementing arXiv:1805.08567 Eq. (39) to HNL_target_flux.root
// (see HNLProduction.hh). Run this from the geant4/ directory, or make
// sure Print.mac / pythia_target_mpi.dat may be written to the CWD.

#include "G4RunManagerFactory.hh"
#include "FTFP_BERT.hh"
#include "G4UImanager.hh"

#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

#include <cstdlib>

int main(int argc, char** argv)
{
  const G4int nProtons = (argc > 1) ? std::atoi(argv[1]) : 1000;

  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);

  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new FTFP_BERT());
  runManager->SetUserInitialization(new ActionInitialization());

  runManager->Initialize();

  G4UImanager* uiManager = G4UImanager::GetUIpointer();
  uiManager->ApplyCommand("/run/verbose 1");
  uiManager->ApplyCommand("/event/verbose 0");
  uiManager->ApplyCommand("/tracking/verbose 0");

  runManager->BeamOn(nProtons);

  delete runManager;
  return 0;
}
