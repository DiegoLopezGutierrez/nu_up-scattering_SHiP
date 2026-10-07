#include "Pythia8VertexModel.hh"
#include "HNLProduction.hh"

#include "Pythia8/Pythia.h"
#include "G4SystemOfUnits.hh"
#include "G4Exception.hh"
#include "Randomize.hh"
#include "G4ios.hh"

#include <array>
#include <cmath>
#include <sstream>

namespace {
  const std::array<G4int, 4> kHeavyFlavorPDG = {411, 431, 521, 541}; // D+, Ds+, B+, Bc+

  // Representative lab energies [GeV] of the proton/neutron pool. The floor is
  // set at 35 GeV lab (sqrt(s_NN) ~ 8.2 GeV), right above ccbar threshold
  const std::array<G4double, 74> kPoolEnergiesGeV =
  {400., 395., 390., 385., 380., 375., 370., 365., 360., 355., 350., 345., 340., 335., 330.,
   325., 320., 315., 310., 305., 300., 295., 290., 285., 280., 275., 270., 265., 260.,
   255., 250., 245., 240., 235., 230., 225., 220., 215., 210., 205., 200., 195., 190.,
   185., 180., 175., 170., 165., 160., 155., 150., 145., 140., 135., 130., 125., 120.,
   115., 110., 105., 100.,  95.,  90.,  85.,  80.,  75.,  70.,  65.,  60.,  55.,  50.,
   45.,  40.,  35.};

  // Coarser pool for the pion/kaon cascade channel (see
  // ../PRODUCTION_RATE_GAP.md, Finding 1: ~35%/23% of total charm/beauty
  // yield is pion/kaon-initiated, per CERN-SHiP-NOTE-2015-009). A 74-point
  // grid here would add 2 species x 2 sub-processes x 74 energies = 296
  // Pythia8 instances at construction time; 15 points keeps the added cost
  // to ~1/5 of that while still resolving the same threshold-suppression
  // shape already seen in the proton/neutron pool.
  const std::array<G4double, 15> kMesonPoolEnergiesGeV =
  {400., 373., 347., 321., 295., 269., 243., 217., 191., 165., 139., 113., 87., 61., 35.};

  // Proton-nucleon inelastic cross section [mb] *as realized inside the Mo
  // target*, not the free proton-proton value (~30 mb at these energies).
  // SteppingAction identifies vertices at the nucleon-level FTFP_BERT
  // inelastic-step granularity inside the nucleus, so the correct
  // denominator is the nuclear-shadowed per-nucleon cross section,
  // sigma_pN = sigma_pA/A (A^0.71 scaling, sub-linear in A) -- not the
  // free-nucleon value. Taken from SHiP TP, arXiv:1504.04956 Sec. 5.3
  // Eq. (5.15) ("sigma_pN = 10.7 mbarn is the hadronic cross section per
  // nucleon in a Mo target"), matching arXiv:1811.00930 Table 1's use of
  // the same number for the same role. See ../PRODUCTION_RATE_GAP.md.
  // Applied uniformly to pion/kaon vertices too, for lack of a dedicated
  // pi-N/K-N inelastic measurement at this granularity -- a documented
  // approximation, not a separately-sourced number.
  constexpr G4double kSigmaInelasticMb = 10.7;

  // Number of events generated at construction time to obtain a stable Pythia cross section
  constexpr int kBurnInEvents = 5000;

  // FTFT K-factors (arXiv:2608.29076, Table 3): separate fits for
  // proton/neutron beams vs. pion/kaon beams.
  constexpr G4double K_charm_pN = 2.48;
  constexpr G4double K_beauty_pN = 1.04;
  constexpr G4double K_charm_piK = 2.02;
  constexpr G4double K_beauty_piK = 1.19;

  G4double KCharmFor(G4int beamPDG) { return (beamPDG == 2212) ? K_charm_pN : K_charm_piK; }
  G4double KBeautyFor(G4int beamPDG) { return (beamPDG == 2212) ? K_beauty_pN : K_beauty_piK; }
}

G4int Pythia8VertexModel::CanonicalBeamPDG(G4int projectilePDG)
{
  const G4int absPdg = std::abs(projectilePDG);
  if (absPdg == 2212 || absPdg == 2112) return 2212; // p, n -> isospin-averaged pool
  if (absPdg == 211) return 211;                     // pi+, pi- -> shared pool (exact C-invariance)
  if (absPdg == 321) return 321;                     // K+, K-  -> shared pool (exact C-invariance)
  return 0;                                          // unhandled (e.g. KS/KL)
}

void Pythia8VertexModel::InitPythiaCCBar(Pythia8::Pythia* pythia, const G4double& eLab, G4int beamPDG) {
  pythia->readString("Beams:frameType = 3");
  pythia->readString("Beams:idA = " + std::to_string(beamPDG));
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

  // FTFT uses the GRV 92 LO pion PDF set for meson beams (no dedicated
  // kaon PDF exists; reused here for K+/- too, same as the FTFT tune
  // itself does not distinguish pi from K at this level).
  if (beamPDG != 2212) {
    pythia->readString("PDF:piSet = 1");
  }

  pythia->readString("HadronLevel:Decay = on");

  pythia->readString("Print:quiet = off");  // on in geant4 model
  pythia->readString("Next:numberCount = 0");
  pythia->readString("Check:epTolErr = 0.1");

  // Cache the (expensive) multiparton-interactions initialization across runs
  std::ostringstream mpiFile;
  mpiFile << "pythia_target_mpi_";
  if (beamPDG != 2212) mpiFile << beamPDG << "_";
  mpiFile << static_cast<int>(eLab) << "GeV.dat";
  pythia->readString("MultipartonInteractions:reuseInit = 3");
  pythia->readString("MultipartonInteractions:initFile = " + mpiFile.str());
}

void Pythia8VertexModel::InitPythiaBBBar(Pythia8::Pythia* pythia, const G4double& eLab, G4int beamPDG) {
  pythia->readString("Beams:frameType = 3");
  pythia->readString("Beams:idA = " + std::to_string(beamPDG));
  pythia->readString("Beams:idB = 2212");
  pythia->settings.parm("Beams:pzA", eLab);
  pythia->readString("Beams:pzB = 0.");

  pythia->readString("HardQCD:hardbbbar = on");

  if (beamPDG != 2212) {
    pythia->readString("PDF:piSet = 1");
  }

  pythia->readString("HadronLevel:Decay = on");

  pythia->readString("Print:quiet = off");  // on in geant4 model
  pythia->readString("Next:numberCount = 0");
  pythia->readString("Check:epTolErr = 0.1");

  // Cache the (expensive) multiparton-interactions initialization across runs
  std::ostringstream mpiFile;
  mpiFile << "pythia_target_mpi_";
  if (beamPDG != 2212) mpiFile << beamPDG << "_";
  mpiFile << static_cast<int>(eLab) << "GeV.dat";
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

void Pythia8VertexModel::BuildSpeciesPool(G4int beamPDG, const G4double* energies, std::size_t nEnergies)
{
  SpeciesPool pool;
  const G4double kCharm = KCharmFor(beamPDG);
  const G4double kBeauty = KBeautyFor(beamPDG);

  for (std::size_t n = 0; n < nEnergies; ++n) {
    const G4double eLab = energies[n];
    pool.energies.push_back(eLab);

    bool ccbar_on = false, bbbar_on = false;

    auto pythiaCC = std::make_unique<Pythia8::Pythia>();
    auto pythiaBB = std::make_unique<Pythia8::Pythia>();

    if (eLab >= 150.0) {
      InitPythiaCCBar(pythiaCC.get(), eLab, beamPDG);
      InitPythiaBBBar(pythiaBB.get(), eLab, beamPDG);
      if (!pythiaCC->init()) {
        G4Exception("Pythia8VertexModel::BuildSpeciesPool", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance (beam "
                                      + std::to_string(beamPDG) + ") at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      if (!pythiaBB->init()) {
        G4Exception("Pythia8VertexModel::BuildSpeciesPool", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance (beam "
                                      + std::to_string(beamPDG) + ") at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      ccbar_on = true;
      bbbar_on = true;
    }
    else if (eLab >= 35.0) {
      InitPythiaCCBar(pythiaCC.get(), eLab, beamPDG);
      if (!pythiaCC->init()) {
        G4Exception("Pythia8VertexModel::BuildSpeciesPool", "Pythia8InitFailed",
                    FatalException, ("Pythia8 failed to initialize pool instance (beam "
                                      + std::to_string(beamPDG) + ") at "
                                      + std::to_string(eLab) + " GeV").c_str());
      }
      ccbar_on = true;
    }

    G4int nCCEvents = 0, nBBEvents = 0;
    G4double nD = 0, nDs = 0, nB = 0, nBc = 0;

    // Burn-in run to obtain a stable cross sections
    for (int i = 0; i < kBurnInEvents; ++i) {
        if ((ccbar_on && !pythiaCC->next()) || (bbbar_on && !pythiaBB->next())) continue;

        nCCEvents++;
        nBBEvents++;

        if (ccbar_on) { ProcessCCBar(pythiaCC.get(), nD, nDs); }

        if (bbbar_on) { ProcessBBBar(pythiaBB.get(), nB, nBc); }
    }

    G4double sigma_ccbar = 0, sigma_bbbar = 0;
    G4double fracD = 0, fracDs = 0, fracB = 0, fracBc = 0;
    G4double sigmaD = 0, sigmaDs = 0, sigmaB = 0, sigmaBc = 0;

    if (ccbar_on && nCCEvents != 0) {
      sigma_ccbar = pythiaCC->info.sigmaGen();
      fracD  = nD / nCCEvents;
      fracDs = nDs / nCCEvents;
      sigmaD  = sigma_ccbar * fracD * kCharm; // FTFT requires rescaling the cross sections by corresponding K-factors
      sigmaDs = sigma_ccbar * fracDs * kCharm; // FTFT requires rescaling the cross sections by corresponding K-factors
    }
    if (bbbar_on && nBBEvents != 0) {
      sigma_bbbar = pythiaBB->info.sigmaGen();
      fracB  = nB / nBBEvents;
      fracBc = nBc / nBBEvents;
      sigmaB  = sigma_bbbar * fracB * kBeauty; // FTFT requires rescaling the cross sections by corresponding K-factors
      sigmaBc = sigma_bbbar * fracBc * kBeauty; // FTFT requires rescaling the cross sections by corresponding K-factors
    }

    pool.sigmaDMb.push_back(sigmaD);
    pool.sigmaDsMb.push_back(sigmaDs);
    pool.sigmaBMb.push_back(sigmaB);
    pool.sigmaBcMb.push_back(sigmaBc);

    // store even if empty to maintain proper indexing
    pool.poolCC.push_back(std::move(pythiaCC));
    pool.poolBB.push_back(std::move(pythiaBB));
  }

  fPools.emplace(beamPDG, std::move(pool));
}

Pythia8VertexModel::Pythia8VertexModel(HNLProduction* hnlProduction, G4int /*targetZ*/, G4int /*targetA*/)
  : fHNLProduction(hnlProduction)
{
  // targetZ/A are accepted for interface documentation purposes but not used
  BuildSpeciesPool(2212, kPoolEnergiesGeV.data(), kPoolEnergiesGeV.size());
  BuildSpeciesPool(211, kMesonPoolEnergiesGeV.data(), kMesonPoolEnergiesGeV.size());
  BuildSpeciesPool(321, kMesonPoolEnergiesGeV.data(), kMesonPoolEnergiesGeV.size());

  // --- Diagnostic (commented out, kept for future use): per-pool-energy
  // cross sections and derived vertex weights, per species, used to track
  // down the production-rate gap vs. arXiv:1811.00930 Table 1+2 (see
  // ../PRODUCTION_RATE_GAP.md). Uncomment to re-enable.
  // G4cout << "\n=== Pythia8VertexModel pool diagnostic ===\n";
  // for (const auto& kv : fPools) {
  //   const G4int beamPDG = kv.first;
  //   const SpeciesPool& pool = kv.second;
  //   G4cout << "-- beamPDG = " << beamPDG << " --\n"
  //          << "E[GeV]  sigmaD[mb]  sigmaDs[mb]  sigmaB[mb]  sigmaBc[mb]  "
  //          << "vtxWeightD  vtxWeightDs  vtxWeightB  vtxWeightBc\n";
  //   for (std::size_t i = 0; i < pool.energies.size(); ++i) {
  //     G4cout << pool.energies[i] << "  "
  //            << pool.sigmaDMb[i] << "  " << pool.sigmaDsMb[i] << "  "
  //            << pool.sigmaBMb[i] << "  " << pool.sigmaBcMb[i] << "  "
  //            << pool.sigmaDMb[i] / kSigmaInelasticMb << "  "
  //            << pool.sigmaDsMb[i] / kSigmaInelasticMb << "  "
  //            << pool.sigmaBMb[i] / kSigmaInelasticMb << "  "
  //            << pool.sigmaBcMb[i] / kSigmaInelasticMb << "\n";
  //   }
  // }
  // G4cout << "===========================================\n" << G4endl;
}

Pythia8VertexModel::~Pythia8VertexModel() = default;

std::size_t Pythia8VertexModel::SelectInstanceIndex(const std::vector<G4double>& energies,
                                                     G4double labEnergyGeV)
{
  std::size_t best = 0;
  G4double bestDiff = std::abs(labEnergyGeV - energies[0]);
  for (std::size_t i = 1; i < energies.size(); ++i) {
    const G4double diff = std::abs(labEnergyGeV - energies[i]);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  }
  return best;
}

void Pythia8VertexModel::ProcessVertex(G4int projectilePDG,
                                        const G4LorentzVector& labMomentum,
                                        const G4ThreeVector& vertexPosition)
{
  const G4int beamPDG = CanonicalBeamPDG(projectilePDG);
  if (beamPDG == 0) return; // unhandled projectile species (e.g. KS/KL)

  auto poolIt = fPools.find(beamPDG);
  if (poolIt == fPools.end()) return; // defensive; should not happen
  SpeciesPool& pool = poolIt->second;

  const G4double labEnergyGeV = labMomentum.e() / CLHEP::GeV;

  // Below the lowest pool energy, treat the charm/beauty yield as zero
  // (see the kPoolEnergiesGeV/kMesonPoolEnergiesGeV comments above).
  if (labEnergyGeV < pool.energies.back()) return;

  const std::size_t idx = SelectInstanceIndex(pool.energies, labEnergyGeV);
  Pythia8::Pythia& pythiaCC = *pool.poolCC[idx];
  Pythia8::Pythia& pythiaBB = *pool.poolBB[idx];

  const G4double vertexWeightD = pool.sigmaDMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightDs = pool.sigmaDsMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightB = pool.sigmaBMb[idx] / kSigmaInelasticMb;
  const G4double vertexWeightBc = pool.sigmaBcMb[idx] / kSigmaInelasticMb;

  if (vertexWeightD == 0.0 && vertexWeightDs == 0.0 && vertexWeightB == 0.0 && vertexWeightBc == 0.0) return;
  if (vertexWeightD < 0.0 || vertexWeightDs < 0.0 || vertexWeightB < 0.0 || vertexWeightBc < 0.0) return;

  // check flags to avoid calling an uninitialized pythia
  bool ccbar_on = false;
  bool bbbar_on = false;

  if (pool.energies[idx] >= 150.0) {
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
