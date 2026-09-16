# HNL production-flux simulation for the SHiP proton target

Geant4 simulation of a 400 GeV proton beam hitting a cylindrical molybdenum
target (radius 125 mm, length 60 cm), estimating the flux of GeV-scale Heavy
Neutral Leptons (HNLs) produced in meson decays at the production point,
following the two-body leptonic decay-width formulas of Bondarenko, Boyarsky,
Gorbunov, Ruchayskiy, *"Phenomenology of GeV-scale Heavy Neutral Leptons"*,
[arXiv:1805.08567](https://arxiv.org/abs/1805.08567), Eq. (39).

## Physics implemented

For every charged pseudoscalar meson h+ = pi+, K+, D+, Ds+, B+, Bc+ produced
in the target, and for a benchmark HNL mass grid (30 log-spaced points from
0.02 to 5 GeV, `HNLProduction::massGrid`), the code computes the two-body
leptonic decay

    h+ -> l_alpha+ + N        (alpha = e, mu, tau)

using

    Gamma(h -> l_a N) = (GF^2 f_h^2 m_h^3)/(8 pi) |V_UD|^2 |U_a|^2
                        x [y_N^2 + y_l^2 - (y_N^2 - y_l^2)^2] sqrt(lambda(1,y_N^2,y_l^2))

(`HNLProduction::GammaPerU2`), divides by the meson's measured total width
(`Gamma_SM = hbar/tau`) to get a branching ratio, and samples the HNL decay
kinematics via an isotropic two-body decay in the meson rest frame boosted
into the lab frame. Because the width is exactly linear in `|U_alpha|^2`,
every output row stores `weightPerU2 = BR(h->l_a N) / |U_alpha|^2` -- a
mixing-independent number -- so the physical flux for any mixing pattern can
be obtained later (in the plotting script) by summing
`weightPerU2 x U_alpha^2` over flavors, without re-running the simulation.

Decay constants, CKM elements, and lifetimes are standard PDG/FLAG values
(`HNLProduction.cc`); the pi -> mu N width formula reproduces the known
SM pi -> mu nu branching ratio (0.977 vs. PDG's 0.9998) in the massless-N
limit, confirming the formula/constants/units.

## Why Pythia8 is needed at all

Geant4's stock hadronic physics lists (FTFP_BERT etc.) do not produce charm
or beauty hadrons -- their string/cascade models simply don't include those
production channels. Since D/Ds/B/Bc leptonic decays are the dominant HNL
production channels above ~1 GeV, a pure-Geant4 simulation would silently
omit them. Architecture:

- **Geant4/FTFP_BERT** (unmodified) transports the 400 GeV proton through
  the target and handles the entire light-hadron (pi/K/p/n) cascade exactly
  as it would for any other application -- this is the part Geant4 is good
  at and there was no reason to touch it.
- **`SteppingAction`** watches every step for a proton/neutron inelastic
  vertex (by process name) and, at that vertex's incident lab energy, calls
  **`Pythia8VertexModel`** purely to obtain the charm/beauty content of that
  specific vertex (Geant4's own secondaries for that same vertex are left
  untouched and continue to be tracked normally -- this is a "riding
  alongside" bookkeeping calculation, not a replacement of Geant4's physics).
- **`TrackingAction`** catches every pi+/-/K+/- track Geant4 creates (any
  generation, from either FTFP_BERT or the Pythia8-driven vertices) for the
  light-meson HNL production channels.
- **`HNLProduction`** implements the width/BR formulas above and writes the
  output ntuple (`HNL_target_flux.root`).

### Pythia8VertexModel implementation notes (read before changing energies/processes)

Getting Pythia8 to supply charm/beauty at an arbitrary, per-vertex energy
turned out to have two hard restrictions that are easy to trip over:

1. `Beams:allowVariableEnergy` (the mechanism Pythia8's own cosmic-ray
   cascade example, `examples/main183.cc`, uses to reuse one instance across
   many collision energies cheaply) **only works with SoftQCD processes** --
   Pythia8 aborts if it is combined with explicit hard processes like
   `HardQCD:hardccbar`. So instead of one variable-energy instance, this
   code keeps a **pool of fully-initialized fixed-energy instances**
   (400, 250, 150, 100, 60, 35 GeV lab) and picks the nearest one per vertex.
2. `SoftQCD:all` and `HardQCD:hardccbar/hardbbbar` **cannot be combined in
   the same instance either** (`"should not combine softQCD processes with
   hard ones"`) -- doing so is numerically unstable (this was found the hard
   way: intermittent segfaults in Pythia8's phase-space sampler, reproducing
   on some runs and not others depending on memory layout). Each pool
   instance therefore runs **only** `HardQCD:hardccbar/hardbbbar`, so every
   generated event is guaranteed to contain a c-cbar or b-bbar pair. This is
   turned back into a physical per-vertex rate with an explicit weight
   `sigma(hard)/sigma(inelastic)`, where `sigma(hard)` comes from Pythia8's
   own `Info::sigmaGen()` (averaged over a 300-event burn-in at construction
   time) and `sigma(inelastic)` is a fixed ~30 mb approximation (good to
   ~20% over this energy range -- replace with a proper energy-dependent
   parameterization if you need better precision). This "biased sampling"
   approach is also far more statistically efficient than waiting for
   charm/beauty to spontaneously appear in inclusive minimum-bias events.

Further approximations (documented in `Pythia8VertexModel.hh`): every
projectile/target nucleon is treated as a proton (isospin symmetry), and the
target nucleus is treated as a free-nucleon gas at rest (no nuclear
shadowing / Fermi motion / Glauber multi-nucleon treatment). Below the
lowest pool energy (35 GeV lab), the charm/beauty yield is treated as zero
(genuinely negligible there, and too close to the ccbar threshold for
Pythia8's sampler to handle reliably).

## Scope / limitations

- Only two-body leptonic channels (h+ -> l+ N) are implemented. Three-body
  semileptonic channels (K -> pi l N, D -> K l N, B -> D l N, etc.) are
  *not* implemented -- they extend the kinematic reach a bit closer to the
  two-body thresholds but are subdominant over most of the SHiP-relevant
  mass range. Neutral mesons have no tree-level charged-current two-body
  leptonic decay to l+N and are correctly excluded.
- Charm/beauty yields rely on the approximations above (isospin-averaged
  free-nucleon target, fixed inelastic cross-section normalization, energy
  pool discretization). Treat the resulting D/Ds/B/Bc-mediated HNL flux as
  an order-of-magnitude/shape estimate, not a precision number -- refine the
  approximations noted above if you need better accuracy.
- No rescattering/absorption of the produced HNL itself is modeled (out of
  scope: this simulation only computes the production flux *at the
  target*, not transport to a downstream detector).

## Building

Requires Geant4 (built with hadronic physics lists) and Pythia8 (locally
verified against Geant4 11.0.1 and Pythia8 8.310).

```
cd geant4
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running

```
./target_sim [nProtons]     # default 1000
```

Produces `HNL_target_flux.root` (ntuple `HNL`, one row per candidate
meson-decay-channel-mass combination) and `n_pot.txt` (number of protons
simulated, for flux normalization). Run from the `build/` directory (or
anywhere with write access) -- Pythia8 will also cache its multiparton-
interactions initialization to `pythia_target_mpi_*GeV.dat` files in the
current directory on first run, speeding up subsequent runs.

Runtime is dominated by the Pythia8 sub-collision calls: expect roughly
0.5 s/proton on a modern laptop core (single-threaded; this simulation
intentionally does not use Geant4 MT, since a shared global Pythia8 pool is
not trivially thread-safe). Scale accordingly for the statistics you need --
given how rare charm/beauty production is per vertex even with the biased
sampling above, a mass-scan flux plot with reasonable statistics in the
D/Ds/B channels will likely want several thousand simulated protons.

A standalone diagnostic, `./test_charm_yield [nVertices]` (default 2000),
exercises `Pythia8VertexModel` directly at a fixed 400 GeV proton-proton
vertex without running a full cascade -- useful for a quick sanity check
after changing the Pythia8 configuration.

### Ntuple columns

| Column | Meaning |
|---|---|
| `mN` | HNL benchmark mass [GeV] |
| `parentPDG` | PDG code of the parent meson |
| `leptonPDG` | PDG code of the accompanying charged lepton |
| `weightPerU2` | BR(h -> l N) / \|U_alpha\|^2 (mixing-independent) |
| `E_N, px_N, py_N, pz_N` | HNL lab 4-momentum [GeV] |
| `E_parent, px_parent, py_parent, pz_parent` | parent meson lab 4-momentum [GeV] |
| `vx, vy, vz` | production vertex [mm] |

See `../analysis/plot_HNL_target_flux.py` for turning this into flux plots
for a chosen HNL mass and flavor-mixing pattern.
