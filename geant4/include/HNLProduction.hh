#ifndef HNLProduction_h
#define HNLProduction_h 1

#include "G4LorentzVector.hh"
#include "G4ThreeVector.hh"
#include <vector>

// ============================================================================
// HNLProduction
//
// Implements HNL production via the two-body leptonic meson decay
//
//     h+ -> l_alpha+ + N
//
// following Eq. (39) of Bondarenko, Boyarsky, Gorbunov, Ruchayskiy,
// "Phenomenology of GeV-scale Heavy Neutral Leptons", arXiv:1805.08567:
//
//     Gamma(h -> l_a N) = (GF^2 f_h^2 m_h^3)/(8 pi) |V_UD|^2 |U_a|^2
//                         x [y_N^2 + y_l^2 - (y_N^2 - y_l^2)^2] sqrt(lambda(1,y_N^2,y_l^2))
//
// with y_l = m_l/m_h, y_N = M_N/m_h and lambda the Kallen function
// (Eq. 14 of the same paper).
//
// The branching ratio is obtained by dividing by the meson's *measured*
// total width (Gamma_SM = hbar/tau), which is an excellent approximation
// since the new-physics partial width above is always a tiny perturbation
// on the total width for the U^2 values of interest.
//
// Because the decay width is exactly linear in |U_alpha|^2, this class
// stores "invariant" branching ratios BR/|U_alpha|^2 in the output ntuple.
// The physical flux for any mixing pattern (Ue2, Umu2, Utau2) can then be
// obtained in the analysis stage (see the companion Python plotting
// script) by summing weight_per_U2 x U_alpha^2 over the three flavors,
// without re-running the (expensive) target simulation.
//
// Scope / limitations
// --------------------
// Only the two-body leptonic channels h+ -> l+ N are implemented, for
// h = pi+, K+, D+, Ds+, B+, Bc+ (all charged pseudoscalar mesons quoted
// in the paper's meson-decay production tables). Neutral mesons have no
// tree-level charged-current 2-body leptonic decay to l+N and are not
// included. Three-body semileptonic channels (e.g. K -> pi l N,
// D -> K l N, B -> D l N) are *not* implemented in this version -- they
// are subdominant to the two-body channels listed here over most of the
// HNL mass range relevant to SHiP, but do extend the kinematic reach
// close to the two-body thresholds. See geant4/README.md.
// ============================================================================

class HNLProduction
{
  public:
    HNLProduction();
    ~HNLProduction() = default;

    // Process one produced meson: for every requested HNL benchmark mass
    // and every kinematically-allowed lepton flavor, sample the HNL decay
    // kinematics and write one row to the analysis ntuple.
    //
    // parentPDG      : PDG code of the parent meson (+-211, +-321, +-411,
    //                   +-431, +-521, +-541)
    // labMomentum    : parent 4-momentum in the lab frame [GeV]
    // vertexPosition : production vertex [mm], stored for reference only
    // vertexWeight   : extra multiplicative weight on top of BR/U^2.
    //                  Used by Pythia8VertexModel for charm/beauty mesons,
    //                  which are generated with a biased (charm/beauty
    //                  mandatory) event sample rather than unbiased
    //                  minimum-bias sampling -- vertexWeight there carries
    //                  sigma(ccbar/bbbar)/sigma(inelastic) so the output
    //                  is still normalized per real inelastic vertex.
    //                  Always 1 for pi/K, which come from actually-
    //                  occurring (unbiased) Geant4 tracks.
    void ProcessMeson(G4int parentPDG,
                       const G4LorentzVector& labMomentum,
                       const G4ThreeVector& vertexPosition,
                       G4double vertexWeight = 1.0);

    // HNL benchmark mass grid [GeV]. Public so PrimaryGeneratorAction /
    // main() can print or override it before the run starts.
    std::vector<G4double> massGrid;

  private:
    // Meson properties needed for the width/BR calculation.
    struct MesonInfo {
      G4int pdg;          // PDG code of the h+ meson
      G4double mass;      // GeV
      G4double fH;        // decay constant [GeV]
      G4double Vckm;      // relevant CKM matrix element
      G4double gammaTotal;// total SM width hbar/tau [GeV]
      const char* name;
    };

    std::vector<MesonInfo> fMesons;

    // Kallen (triangle) function.
    static G4double Lambda(G4double a, G4double b, G4double c);

    // Gamma(h -> l N) / |U_alpha|^2, Eq. (39) of 1805.08567. Returns 0 if
    // kinematically forbidden (m_h < m_l + m_N).
    static G4double GammaPerU2(const MesonInfo& h, G4double mLepton, G4double mN);

    // Sample an isotropic two-body decay h -> l N in the meson rest frame
    // and boost the HNL 4-momentum into the lab frame.
    G4LorentzVector SampleHNLKinematics(const G4LorentzVector& labMomentum,
                                         G4double mH, G4double mLepton, G4double mN) const;

    const MesonInfo* FindMeson(G4int pdg) const;
};

#endif
