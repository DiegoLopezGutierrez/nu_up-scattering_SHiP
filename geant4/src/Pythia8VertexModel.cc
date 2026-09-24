#include "Pythia8VertexModel.hh"
#include "HNLProduction.hh"

#include "Pythia8/Pythia.h"
#include "G4SystemOfUnits.hh"
#include "G4Exception.hh"
#include "Randomize.hh"

#include <array>
#include <cmath>
#include <sstream>

namespace {
  const std::array<G4int, 4> kHeavyFlavorPDG = {411, 431, 521, 541}; // D+, Ds+, B+, Bc+

  // Representative lab energies [GeV] of the pool. The floor is
  // set at 35 GeV lab (sqrt(s_NN) ~ 8.2 GeV), right above ccbar threshold
  const std::array<G4double, 15> kPoolEnergiesGeV = 
      {400., 373., 347., 321., 295., 269., 243., 217., 191., 165., 139., 113., 87., 61., 35.};

  // Fixed approximation for the total proton-proton inelastic cross section [mb]
  constexpr G4double kSigmaInelasticMb = 30.0;

  // Number of events generated at construction time to obtain a stable Pythia cross section
  constexpr int kBurnInEvents = 5000;

  constexpr G4double K_charm = 2.48; // FTFT, K-factor for charm from p/n beam
  constexpr G4double K_beauty = 1.04; // FTFT, K-factor for beauty from p/n beam
}

void Pythia8VertexModel::InitPythiaCCBar(Pythia8::Pythia* pythia, const G4double& eLab) {
  pythia->readString("Beams:frameType = 3");
  pythia->readString("Beams:idA = 2212");
  pythia->readString("Beams:idB = 2212");
  pythia->settings.parm("Beams:pzA", eLab);
  pythia->readString("Beams:pzB = 0.");

  pythia->readString("HardQCD:hardccbar = on");
  pythia->readString("StringZ:aLund = 2.0");
  pythia->readString("StringZ:bLund = 0.2");
  pythia->readString("StringZ:rFactC = 2.0");

  pythia->readString("MultipartonInteractions:ecmRef = 30");
  pythia->readString("MultipartonInteractions:pT0Ref = 0.69");
  pythia->readString("MultipartonInteractions:ecmPow = 0.266");

  pythia->readString("BeamRemnants:halfMassForKT = 1.21");

  pythia->readString("HadronLevel:Decay = on");

  pythia->readString("Print:quiet = off");  // on in geant4 model
  pythia->readString("Next:numberCount = 0");
  pythia->readString("Check:epTolErr = 0.1");

  // Cache the (expensive) multiparton-interactions initialization across runs
  std::ostringstream mpiFile;
  mpiFile << "pythia_target_mpi_" << static_cast<int>(eLab) << "GeV.dat";
  pythia->readString("MultipartonInteractions:reuseInit = 3");
  pythia->readString("MultipartonInteractions:initFile = " + mpiFile.str());
}

void Pythia8VertexModel::InitPythiaBBBar(Pythia8::Pythia* pythia, const G4double& eLab) {
  pythia->readString("Beams:frameType = 3");
  pythia->readString("Beams:idA = 2212");
  pythia->readString("Beams:idB = 2212");
  pythia->settings.parm("Beams:pzA", eLab);
  pythia->readString("Beams:pzB = 0.");

  pythia->readString("HardQCD:hardbbbar = on");

  pythia->readString("HadronLevel:Decay = on");

  pythia->readString("Print:quiet = off");  // on in geant4 model
  pythia->readString("Next:numberCount = 0");
  pythia->readString("Check:epTolErr = 0.1");

  // Cache the (expensive) multiparton-interactions initialization across runs
  std::ostringstream mpiFile;
  mpiFile << "pythia_target_mpi_" << static_cast<int>(eLab) << "GeV.dat";
  pythia->readString("MultipartonInteractions:reuseInit = 3");
  pythia->readString("MultipartonInteractions:initFile = " + mpiFile.str());
}

void Pythia8VertexModel::ProcessCCBar(Pythia8::Pythia* pythia, G4double& nD, G4double& nDs) {
  const Pythia8::Event& event = pythia->event;
  for (int j = 1; j < event.size(); ++j) { // iterates over all particles in event
      const int absId = std::abs(event[j].id());
      bool isHeavyFlavor = false;

      for (const auto& pdg : kHeavyFlavorPDG) {
          if (absId == pdg) { 
              isHeavyFlavor = true; 
              if (absId == 411) {
                  int d1 = event[j].daughter1();
                  if (abs(event[d1].id()) != 411) {
                      nD++;
                  }
              }
              if (absId == 431) {
                  int d1 = event[j].daughter1();
                  if (abs(event[d1].id()) != 431) {
                      nDs++;
                  }
              }
              break;
          }
      }
      if (!isHeavyFlavor) continue;
  }
}

void Pythia8VertexModel::ProcessBBBar(Pythia8::Pythia* pythia, G4double& nB, G4double& nBc) {
  const Pythia8::Event& event = pythia->event;
  for (int j = 1; j < event.size(); ++j) { // iterates over all particles in event
      const int absId = std::abs(event[j].id());
      bool isHeavyFlavor = false;

      for (const auto& pdg : kHeavyFlavorPDG) {
          if (absId == pdg) { 
              isHeavyFlavor = true; 
              if (absId == 521) {
                  int d1 = event[j].daughter1();
                  if (abs(event[d1].id()) != 521) {
                      nB++;
                  }
              }
              if (absId == 541) {
                  int d1 = event[j].daughter1();
                  if (abs(event[d1].id()) != 541) {
                      nBc++;
                  }
              }
              break;
          }
      }
      if (!isHeavyFlavor) continue;
  }
}

Pythia8VertexModel::Pythia8VertexModel(HNLProduction* hnlProduction, G4int /*targetZ*/, G4int /*targetA*/)
  : fHNLProduction(hnlProduction)
{
  // targetZ/A are accepted for interface documentation purposes but not used
  for (const auto& eLab : kPoolEnergiesGeV) {
    fPoolEnergies.push_back(eLab);

    bool ccbar_on = false, bbbar_on = false;

    auto pythiaCC = std::make_unique<Pythia8::Pythia>();
    auto pythiaBB = std::make_unique<Pythia8::Pythia>();

    if (eLab >= 150.0) {
      InitPythiaCCBar(pythiaCC.get(), eLab);
      InitPythiaBBBar(pythiaBB.get(), eLab);
      if (!pythiaCC->init()) {
        G4Exception("Pythia8VertexModel::Pythia8VertexModel", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      if (!pythiaBB->init()) {
        G4Exception("Pythia8VertexModel::Pythia8VertexModel", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      ccbar_on = true;
      bbbar_on = true;
    }
    else if (eLab >= 35.0) {
      InitPythiaCCBar(pythiaCC.get(), eLab);
      if (!pythiaCC->init()) {
        G4Exception("Pythia8VertexModel::Pythia8VertexModel", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      ccbar_on = true;
    }

    // // nucleon-nucleus collision modelled as proton-proton with fixed
    // // incoming proton energy.
    // pythia->readString("Beams:frameType = 3");
    // pythia->readString("Beams:idA = 2212");
    // pythia->readString("Beams:idB = 2212");
    // pythia->settings.parm("Beams:pzA", eLab);
    // pythia->readString("Beams:pzB = 0.");

    // // Explicit hard-QCD charm/beauty pair production only 
    // pythia->readString("HardQCD:hardccbar = on");
    // pythia->readString("HardQCD:hardbbbar = on");

    // pythia->readString("HadronLevel:Decay = on");

    // pythia->readString("Print:quiet = on");
    // pythia->readString("Next:numberCount = 0");
    // pythia->readString("Check:epTolErr = 0.1");

    // // Implement FTFT tune from 2608.29076
    // // note: technically, bbbar should only be modified with the K-factor but
    // // testing shows that the fragmentation and MPI parameters do not affect
    // // bbbar yields enough to warrant a separate Pythia instance.
    // pythia->readString("StringZ:aLund = 2.0");
    // pythia->readString("StringZ:bLund = 0.2");
    // pythia->readString("StringZ:rFactC = 2.0");

    // pythia->readString("MultipartonInteractions:ecmRef = 30");
    // pythia->readString("MultipartonInteractions:pT0Ref = 0.69");
    // pythia->readString("MultipartonInteractions:ecmPow = 0.266");

    // pythia->readString("BeamRemnants:halfMassForKT = 1.21");

    // Cache the (expensive) multiparton-interactions initialization across runs
    // std::ostringstream mpiFile;
    // mpiFile << "pythia_target_mpi_" << static_cast<int>(eLab) << "GeV.dat";
    // pythia->readString("MultipartonInteractions:reuseInit = 3");
    // pythia->readString("MultipartonInteractions:initFile = " + mpiFile.str());

    // if (!pythia->init()) {
    //   G4Exception("Pythia8VertexModel::Pythia8VertexModel", "Pythia8InitFailed",
    //                FatalException, ("Pythia8 failed to initialize pool instance at "
    //                                  + std::to_string(eLab) + " GeV").c_str());
    // }

    G4int nCCEvents = 0, nBBEvents = 0;
    G4double nD = 0, nDs = 0, nB = 0, nBc = 0;

    // Burn-in run to obtain a stable sigma(ccbar+bbbar) estimate; the
    // events themselves are discarded.
    for (int i = 0; i < kBurnInEvents; ++i) {
        if ((ccbar_on && !pythiaCC->next()) || (bbbar_on && !pythiaBB->next())) continue;

        nCCEvents++;
        nBBEvents++;

        if (ccbar_on) {
          ProcessCCBar(pythiaCC.get(), nD, nDs);
            // const Pythia8::Event& eventCC = pythiaCC->event;
            // for (int j = 1; j < eventCC.size(); ++j) { // iterates over all particles in event
            //     const int absId = std::abs(eventCC[j].id());
            //     bool isHeavyFlavor = false;
            //     std::vector<double> particle;
            //     std::string name;

            //     for (const auto& pdg : kHeavyFlavorPDG) {
            //         if (absId == pdg) { 
            //             isHeavyFlavor = true; 
            //             name = pdg_to_str[eventCC[j].id()];
            //             if (absId == 411) {
            //                 int d1 = eventCC[j].daughter1();
            //                 if (abs(eventCC[d1].id()) != 411) {
            //                     Dpmcounter++;
            //                 }
            //             }
            //             if (absId == 431) {
            //                 int d1 = eventCC[j].daughter1();
            //                 if (abs(eventCC[d1].id()) != 431) {
            //                     Dspmcounter++;
            //                 }
            //             }
            //             break;
            //         }
            //     }
            //     if (!isHeavyFlavor) continue;
            // }
        }

        if (bbbar_on) {
          ProcessBBBar(pythiaBB.get(), nB, nBc);
            // const Pythia8::Event& eventBB = pythiaBB->event;
            // for (int j = 1; j < eventBB.size(); ++j) { // iterates over all particles in event
            //     const int absId = std::abs(eventBB[j].id());
            //     bool isHeavyFlavor = false;
            //     std::vector<double> particle;
            //     std::string name;

            //     for (const auto& pdg : kHeavyFlavorPDG) {
            //         if (absId == pdg) { 
            //             isHeavyFlavor = true; 
            //             name = pdg_to_str[eventBB[j].id()];
            //             if (absId == 521) {
            //                 int d1 = eventBB[j].daughter1();
            //                 if (abs(eventBB[d1].id()) != 521) {
            //                     Bpmcounter++;
            //                 }
            //             }
            //             if (absId == 541) {
            //                 int d1 = eventBB[j].daughter1();
            //                 if (abs(eventBB[d1].id()) != 541) {
            //                     Bcpmcounter++;
            //                 }
            //             }
            //             break;
            //         }
            //     }
            //     if (!isHeavyFlavor) continue;
            // }
        }
    }

    G4double sigma_ccbar = 0, sigma_bbbar = 0;
    G4double fracD = 0, fracDs = 0, fracB = 0, fracBc = 0;
    G4double sigmaD = 0, sigmaDs = 0, sigmaB = 0, sigmaBc = 0;

    if (ccbar_on && nCCEvents != 0) {
      sigma_ccbar = pythiaCC->info.sigmaGen();
      fracD  = nD / nCCEvents;
      fracDs = nDs / nCCEvents;
      sigmaD  = sigma_ccbar * fracD * K_charm; // FTFT requires rescaling the cross sections by corresponding K-factors
      sigmaDs = sigma_ccbar * fracDs * K_charm; // FTFT requires rescaling the cross sections by corresponding K-factors
    }
    if (bbbar_on && nBBEvents != 0) {
      sigma_bbbar = pythiaBB->info.sigmaGen();
      fracB  = nB / nBBEvents;
      fracBc = nBc / nBBEvents;
      sigmaB  = sigma_bbbar * fracB * K_beauty; // FTFT requires rescaling the cross sections by corresponding K-factors
      sigmaBc = sigma_bbbar * fracBc * K_beauty; // FTFT requires rescaling the cross sections by corresponding K-factors
    }

    // if ((ccFlag) && (Dpmcounter != 0 || Dspmcounter != 0)) {
    //     // fracD  = Dpmcounter  / double(Dpmcounter + Dspmcounter);
    //     fracD  = Dpmcounter / nCCEvents;
    //     // fracDs = 1.0 - fracD;
    //     fracDs = Dspmcounter / nCCEvents;
    //     sigmaD  = sigma_cc * fracD * K_charm; // FTFT requires rescaling the cross sections by corresponding K-factors
    //     sigmaDs = sigma_cc * fracDs * K_charm; // FTFT requires rescaling the cross sections by corresponding K-factors

    //     // calculate yields
    //     yieldD = sigmaD / kSigmaInelasticMb;
    //     yieldDs = sigmaDs / kSigmaInelasticMb;
    // }

    // if ((bbFlag) && (Bpmcounter != 0 || Bcpmcounter != 0)) {
    //     // fracB  = Bpmcounter  / double(Bpmcounter + Bcpmcounter);   // still noisy at low stats, but a ratio, not an absolute rate
    //     // fracBc = 1.0 - fracB;
    //     fracB  = Bpmcounter  / nBBEvents;   // still noisy at low stats, but a ratio, not an absolute rate
    //     fracBc = Bcpmcounter / nBBEvents;
    //     sigmaB  = sigma_bb * fracB * K_beauty; // FTFT requires rescaling the cross sections by corresponding K-factors
    //     sigmaBc = sigma_bb * fracBc * K_beauty; // FTFT requires rescaling the cross sections by corresponding K-factors

    //     // calculate yields
    //     yieldB = sigmaB / kSigmaInelasticMb;
    //     yieldBc = sigmaBc / kSigmaInelasticMb;
    // }

    // if (nD != 0 || nDs != 0) {
    //     fracD  = nD  / double(nD + nDs);
    //     fracDs = 1.0 - fracD;
    //     sigmaD  = sigma_ccbar * fracD * K_charm;
    //     sigmaDs = sigma_ccbar * fracDs * K_charm;
    // }

    // if (nB != 0 || nBc != 0) {
    //     fracB  = nB  / double(nB + nBc);
    //     fracBc = 1.0 - fracB;
    //     sigmaB  = sigma_bbbar * fracB * K_beauty;
    //     sigmaBc = sigma_bbbar * fracBc * K_beauty;
    // }
    // fSigmaHardMb.push_back(pythia->info.sigmaGen());
    fSigmaDMb.push_back(sigmaD);
    fSigmaDsMb.push_back(sigmaDs);
    fSigmaBMb.push_back(sigmaB);
    fSigmaBcMb.push_back(sigmaBc);

    // store even if empty to maintain proper indexing
    fPoolCC.push_back(std::move(pythiaCC));
    fPoolBB.push_back(std::move(pythiaBB));
  }
}

Pythia8VertexModel::~Pythia8VertexModel() = default;

std::size_t Pythia8VertexModel::SelectInstanceIndex(G4double labEnergyGeV) const
{
  std::size_t best = 0;
  G4double bestDiff = std::abs(labEnergyGeV - fPoolEnergies[0]);
  for (std::size_t i = 1; i < fPoolEnergies.size(); ++i) {
    const G4double diff = std::abs(labEnergyGeV - fPoolEnergies[i]);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  }
  return best;
}

void Pythia8VertexModel::ProcessVertex(G4int /*projectilePDG*/,
                                        const G4LorentzVector& labMomentum,
                                        const G4ThreeVector& vertexPosition)
{
  const G4double labEnergyGeV = labMomentum.e() / CLHEP::GeV;

  // Below the lowest pool energy, treat the charm/beauty yield as zero
  // (see the kPoolEnergiesGeV comment above).
  if (labEnergyGeV < fPoolEnergies.back()) return;

  const std::size_t idx = SelectInstanceIndex(labEnergyGeV);
  Pythia8::Pythia& pythiaCC = *fPoolCC[idx];
  Pythia8::Pythia& pythiaBB = *fPoolBB[idx];

  // const G4double vertexWeight = fSigmaHardMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightD = fSigmaDMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightDs = fSigmaDsMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightB = fSigmaBMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightBc = fSigmaBcMb[idx] / kSigmaInelasticMb;

  // if (vertexWeight <= 0.0) return;
  if (vertexWeightD == 0.0 && vertexWeightDs == 0.0 && vertexWeightB == 0.0 && vertexWeightBc == 0.0) return;
  if (vertexWeightD < 0.0 || vertexWeightDs < 0.0 || vertexWeightB < 0.0 || vertexWeightBc < 0.0) return;

  // check flags to avoid calling an uninitialized pythia
  bool ccbar_on = false;
  bool bbbar_on = false;

  if (fPoolEnergies[idx] >= 150.0) {
    ccbar_on = true;
    bbbar_on = true;
  }
  else {
    ccbar_on = true;
  }

  // try up to 10 times to get an event
  const G4double nTries = 10;
  if (ccbar_on) {
    for (int k = 1; k < nTries; ++k) {
      if (pythiaCC.next()) break; // we just need the first event that works
    }
  }
  if (bbbar_on) {
    for (int k = 1; k < nTries; ++k) {
      if (pythiaBB.next()) break; // we just need the first event that works
    }
  }

  const G4ThreeVector labDirection = labMomentum.vect().unit();

  if (ccbar_on) {
    const Pythia8::Event& eventCC = pythiaCC.event;
    for (int i = 1; i < eventCC.size(); ++i) {
      const int absId = std::abs(eventCC[i].id());
      bool isHeavyFlavor = false;
      for (const auto& pdg : kHeavyFlavorPDG) {
        if (absId == pdg) { isHeavyFlavor = true; break; }
      }
      if (!isHeavyFlavor) continue;

      const Pythia8::Vec4 p = eventCC[i].p();

      // rotate Pythia's z-aligned beam to lab frame
      G4ThreeVector p3(p.px(), p.py(), p.pz());
      p3.rotateUz(labDirection);

      const G4LorentzVector labP4(p3 * CLHEP::GeV, p.e() * CLHEP::GeV);
      if (absId == 411) {
        fHNLProduction->ProcessMeson(eventCC[i].id(), labP4, vertexPosition, vertexWeightD);
      }
      if (absId == 431) {
        fHNLProduction->ProcessMeson(eventCC[i].id(), labP4, vertexPosition, vertexWeightDs);
      }
    }
  }

  if (bbbar_on) {
    const Pythia8::Event& eventBB = pythiaBB.event;
    for (int i = 1; i < eventBB.size(); ++i) { // iterates over all particles in event
      const int absId = std::abs(eventBB[i].id());
      bool isHeavyFlavor = false;
      for (const auto& pdg : kHeavyFlavorPDG) {
        if (absId == pdg) { isHeavyFlavor = true; break; }
      }
      if (!isHeavyFlavor) continue;

      const Pythia8::Vec4 p = eventBB[i].p();

      // rotate Pythia's z-aligned beam to lab frame
      G4ThreeVector p3(p.px(), p.py(), p.pz());
      p3.rotateUz(labDirection);

      const G4LorentzVector labP4(p3 * CLHEP::GeV, p.e() * CLHEP::GeV);

      if (absId == 521) {
        fHNLProduction->ProcessMeson(eventBB[i].id(), labP4, vertexPosition, vertexWeightB);
      }
      if (absId == 541) {
        fHNLProduction->ProcessMeson(eventBB[i].id(), labP4, vertexPosition, vertexWeightBc);
      }
    }
  }
}
