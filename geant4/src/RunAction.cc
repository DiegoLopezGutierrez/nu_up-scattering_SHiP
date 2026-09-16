#include "RunAction.hh"

#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"

#include <fstream>

RunAction::RunAction()
{
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->SetVerboseLevel(1);
  analysisManager->SetNtupleMerging(true);

  analysisManager->CreateNtuple("HNL", "HNL production kinematics per candidate meson decay");
  analysisManager->CreateNtupleDColumn("mN");          // HNL benchmark mass [GeV]
  analysisManager->CreateNtupleIColumn("parentPDG");   // parent meson PDG code
  analysisManager->CreateNtupleIColumn("leptonPDG");   // charged-lepton PDG code in h -> l N
  analysisManager->CreateNtupleDColumn("weightPerU2"); // BR(h->lN) / |U_alpha|^2
  analysisManager->CreateNtupleDColumn("E_N");         // HNL lab energy [GeV]
  analysisManager->CreateNtupleDColumn("px_N");
  analysisManager->CreateNtupleDColumn("py_N");
  analysisManager->CreateNtupleDColumn("pz_N");
  analysisManager->CreateNtupleDColumn("E_parent");    // parent meson lab energy [GeV]
  analysisManager->CreateNtupleDColumn("px_parent");
  analysisManager->CreateNtupleDColumn("py_parent");
  analysisManager->CreateNtupleDColumn("pz_parent");
  analysisManager->CreateNtupleDColumn("vx");          // production vertex [mm]
  analysisManager->CreateNtupleDColumn("vy");
  analysisManager->CreateNtupleDColumn("vz");
  analysisManager->FinishNtuple();
}

void RunAction::BeginOfRunAction(const G4Run* /*run*/)
{
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->OpenFile("HNL_target_flux.root");
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->Write();
  analysisManager->CloseFile();

  // Number of simulated protons-on-target, needed by the plotting script
  // to normalize the HNL flux per POT.
  std::ofstream pot("n_pot.txt");
  pot << run->GetNumberOfEventToBeProcessed() << "\n";
}
