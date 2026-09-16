// Standalone diagnostic (not part of the target simulation itself):
// fires the Pythia8VertexModel directly, many times, at a fixed 400 GeV
// proton-on-nucleon vertex, to measure the charm/beauty yield per vertex
// without waiting for a full Geant4 cascade run. Useful to sanity-check
// that HardQCD:hardccbar/hardbbbar are actually firing and being
// harvested correctly.
//
// Usage: ./test_charm_yield [nVertices]

#include "Pythia8VertexModel.hh"
#include "HNLProduction.hh"
#include "G4LorentzVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
  const int nVertices = (argc > 1) ? std::atoi(argv[1]) : 2000;

  // HNLProduction fills its rows through the global G4AnalysisManager
  // singleton, same as in the full simulation; book/open it here since
  // this standalone diagnostic does not go through RunAction.
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->CreateNtuple("HNL", "HNL production kinematics per candidate meson decay");
  analysisManager->CreateNtupleDColumn("mN");
  analysisManager->CreateNtupleIColumn("parentPDG");
  analysisManager->CreateNtupleIColumn("leptonPDG");
  analysisManager->CreateNtupleDColumn("weightPerU2");
  analysisManager->CreateNtupleDColumn("E_N");
  analysisManager->CreateNtupleDColumn("px_N");
  analysisManager->CreateNtupleDColumn("py_N");
  analysisManager->CreateNtupleDColumn("pz_N");
  analysisManager->CreateNtupleDColumn("E_parent");
  analysisManager->CreateNtupleDColumn("px_parent");
  analysisManager->CreateNtupleDColumn("py_parent");
  analysisManager->CreateNtupleDColumn("pz_parent");
  analysisManager->CreateNtupleDColumn("vx");
  analysisManager->CreateNtupleDColumn("vy");
  analysisManager->CreateNtupleDColumn("vz");
  analysisManager->FinishNtuple();
  analysisManager->OpenFile("test_charm_yield.root");

  HNLProduction hnl;
  Pythia8VertexModel model(&hnl, 42, 96);

  const G4LorentzVector proton400(0.0, 0.0, 400.0 * GeV,
                                   std::sqrt(400.0 * 400.0 + 0.938 * 0.938) * GeV);

  for (int i = 0; i < nVertices; ++i) {
    model.ProcessVertex(2212, proton400, G4ThreeVector());
  }

  analysisManager->Write();
  analysisManager->CloseFile();

  std::cout << "Ran " << nVertices
            << " proton-nucleon vertices at 400 GeV. "
            << "See test_charm_yield.root for any D+/Ds+/B+/Bc+ -> l N rows "
            << "(parentPDG in {411,431,521,541})." << std::endl;

  return 0;
}
