#include "HNLProduction.hh"

#include "G4AnalysisManager.hh"
#include "Randomize.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <array>

namespace {
  // Fermi constant [GeV^-2] -- same value used throughout analysis/HNLClass.py
  constexpr G4double kGF = 1.1663787e-5;

  // hbar in GeV*s, used to convert PDG lifetimes to total widths.
  constexpr G4double kHbar = 6.582119569e-25;

  // Charged-lepton masses [GeV] (PDG).
  struct LeptonInfo { G4int pdg; G4double mass; };
  const std::array<LeptonInfo, 3> kLeptons = {{
    {11, 0.000510998950},  // e-
    {13, 0.1056583755},    // mu-
    {15, 1.77686}          // tau-
  }};
}

HNLProduction::HNLProduction()
{
  // Default HNL benchmark mass grid [GeV]: log-spaced, spanning from just
  // above nothing up to the B+ -> tau+ N threshold. Edit freely -- this
  // list is re-used for every produced meson, it does not require
  // re-running the (expensive) Geant4/Pythia8 cascade.
  const int nPoints = 30;
  const G4double mMin = 0.02, mMax = 5.0;
  for (int i = 0; i < nPoints; ++i) {
    G4double logM = std::log(mMin) + i * (std::log(mMax) - std::log(mMin)) / (nPoints - 1);
    massGrid.push_back(std::exp(logM));
  }

  // Meson inputs. Decay constants and CKM elements are the standard
  // PDG/FLAG values used throughout the HNL-phenomenology literature
  // (consistent with Table 8 / App. C of arXiv:1805.08567); total widths
  // are obtained from the measured PDG lifetimes via Gamma = hbar/tau.
  //
  //   meson   quark content   CKM element
  //   pi+     u dbar          Vud
  //   K+      u sbar          Vus
  //   D+      c dbar          Vcd
  //   Ds+     c sbar          Vcs
  //   B+      u bbar          Vub
  //   Bc+     c bbar          Vcb
  fMesons = {
    { 211, 0.13957039, 0.1302, 0.97370, kHbar / 2.6033e-8,  "pi+" },
    { 321, 0.493677,   0.1556, 0.2245,  kHbar / 1.2380e-8,  "K+"  },
    { 411, 1.86966,    0.2120, 0.221,   kHbar / 1.033e-12,  "D+"  },
    { 431, 1.96835,    0.2499, 0.975,   kHbar / 5.04e-13,   "Ds+" },
    { 521, 5.27934,    0.1871, 0.00382, kHbar / 1.638e-12,  "B+"  },
    { 541, 6.2749,     0.434,  0.0410,  kHbar / 5.10e-13,   "Bc+" },
  };
}

const HNLProduction::MesonInfo* HNLProduction::FindMeson(G4int pdg) const
{
  G4int absPdg = std::abs(pdg);
  for (const auto& meson : fMesons) {
    if (meson.pdg == absPdg) return &meson;
  }
  return nullptr;
}

G4double HNLProduction::Lambda(G4double a, G4double b, G4double c)
{
  return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c;
}

G4double HNLProduction::GammaPerU2(const MesonInfo& h, G4double mLepton, G4double mN)
{
  if (h.mass <= mLepton + mN) return 0.0;

  const G4double yL = mLepton / h.mass;
  const G4double yN = mN / h.mass;

  const G4double lam = Lambda(1.0, yN * yN, yL * yL);
  if (lam <= 0.0) return 0.0;

  const G4double bracket = yN * yN + yL * yL - (yN * yN - yL * yL) * (yN * yN - yL * yL);

  return (kGF * kGF * h.fH * h.fH * h.mass * h.mass * h.mass) / (8.0 * pi)
         * h.Vckm * h.Vckm
         * bracket * std::sqrt(lam);
}

G4LorentzVector HNLProduction::SampleHNLKinematics(const G4LorentzVector& labMomentum,
                                                    G4double mH, G4double mLepton, G4double mN) const
{
  // Two-body phase space in the meson rest frame: |p*| from the Kallen
  // function, isotropic direction (pseudoscalar meson => no residual
  // polarization correlation for this decay).
  const G4double lam = Lambda(mH * mH, mLepton * mLepton, mN * mN);
  const G4double pStar = std::sqrt(std::max(lam, 0.0)) / (2.0 * mH);
  const G4double eStarN = std::sqrt(pStar * pStar + mN * mN);

  const G4double cosTheta = 2.0 * G4UniformRand() - 1.0;
  const G4double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
  const G4double phi = CLHEP::twopi * G4UniformRand();

  G4LorentzVector pN_rest(pStar * sinTheta * std::cos(phi),
                           pStar * sinTheta * std::sin(phi),
                           pStar * cosTheta,
                           eStarN);

  // Boost from the meson rest frame into the lab frame.
  const G4ThreeVector beta = labMomentum.boostVector();
  pN_rest.boost(beta);

  return pN_rest;
}

void HNLProduction::ProcessMeson(G4int parentPDG,
                                  const G4LorentzVector& labMomentum,
                                  const G4ThreeVector& vertexPosition,
                                  G4double vertexWeight)
{
  const MesonInfo* h = FindMeson(parentPDG);
  if (!h) return;

  auto analysisManager = G4AnalysisManager::Instance();

  for (const auto& mN : massGrid) {
    for (const auto& lep : kLeptons) {

      const G4double gammaPerU2 = GammaPerU2(*h, lep.mass, mN);
      if (gammaPerU2 <= 0.0) continue;

      const G4double brPerU2 = (gammaPerU2 / h->gammaTotal) * vertexWeight;

      const G4LorentzVector pN = SampleHNLKinematics(labMomentum, h->mass, lep.mass, mN);

      G4int col = 0;
      analysisManager->FillNtupleDColumn(col++, mN);
      analysisManager->FillNtupleIColumn(col++, parentPDG);
      analysisManager->FillNtupleIColumn(col++, lep.pdg);
      analysisManager->FillNtupleDColumn(col++, brPerU2);
      analysisManager->FillNtupleDColumn(col++, pN.e());
      analysisManager->FillNtupleDColumn(col++, pN.px());
      analysisManager->FillNtupleDColumn(col++, pN.py());
      analysisManager->FillNtupleDColumn(col++, pN.pz());
      // labMomentum is in Geant4 internal units (MeV); pN above is a raw
      // GeV-valued number (all HNLProduction physics constants are plain
      // GeV numbers, not G4-unit-scaled), so convert here for consistency.
      analysisManager->FillNtupleDColumn(col++, labMomentum.e() / CLHEP::GeV);
      analysisManager->FillNtupleDColumn(col++, labMomentum.px() / CLHEP::GeV);
      analysisManager->FillNtupleDColumn(col++, labMomentum.py() / CLHEP::GeV);
      analysisManager->FillNtupleDColumn(col++, labMomentum.pz() / CLHEP::GeV);
      analysisManager->FillNtupleDColumn(col++, vertexPosition.x() / CLHEP::mm);
      analysisManager->FillNtupleDColumn(col++, vertexPosition.y() / CLHEP::mm);
      analysisManager->FillNtupleDColumn(col++, vertexPosition.z() / CLHEP::mm);
      analysisManager->AddNtupleRow();
    }
  }
}
