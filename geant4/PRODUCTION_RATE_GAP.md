# HNL production-rate normalization gap: investigation notes

## Status: two fixes applied, ~12x still open

`kSigmaInelasticMb` corrected 30.0 -> 10.7 mb (Finding 2): gap
**109.7x -> 39.1x** (300-proton run, Ds+ -> N+e, Ue2=1), confirming the
diagnosis. The pion/kaon cascade channel (Finding 1) is now implemented in
`Pythia8VertexModel`/`SteppingAction` and empirically measured at **+11.3%**
heavy-flavor flux: gap **39.1x -> 35.1x**. Combined with the still-open
~2.9x raw cross-section gap (Finding 3, not yet fixed, quantified as a
candidate only): known factors explain ~3.2x of the original 39.1x.
**~12x remains unexplained.**

## Summary

Validating our simulated charm-mediated HNL flux (e.g. Ds+ -> N+e, Ue2=1)
against the values implied by SHiP collaboration, arXiv:1811.00930 (Table 1,
Table 2, Fig. 2) found our per-POT yield **~105-110x lower**, then **~39x**
after the `kSigmaInelasticMb` fix, then **~35x** after implementing the
pion/kaon cascade channel. Of the original 39.1x, **~3.2x is explained** by
two confirmed effects (one applied, one still just quantified) and **~12x is
still open**. This note records the investigation: what's ruled out, what's
fixed, what's explained, and what's still open.

## Method

- Instrumented `SteppingAction` to count cascade inelastic vertices per POT,
  split by nucleon- vs. meson-initiated, and how many clear the 35 GeV
  charm-production floor.
- Instrumented `Pythia8VertexModel`'s constructor to print the per-pool-
  energy, K-factor-corrected cross sections and derived vertex weights, per
  beam species.
- Cross-checked our simulated per-POT Ds+ yield against an independent
  Python re-implementation of `HNLProduction::GammaPerU2`, to separate the
  production-rate normalization from the decay branching-ratio physics.
- Ran `./target_sim` at 200 and 300 protons for stable statistics -- the
  originally-quoted ~264x gap came from a 10-proton test file and was
  itself statistics-limited (103x at 200 protons, 110x at 300).
- To isolate the pion/kaon channel's effect once implemented: ran matched
  300-proton pairs with `SteppingAction::IsCascadeBeamSpecies` restricted to
  nucleon-only vs. the full p/n/pi+-/K+- set, same binary otherwise, and
  compared `flux_vs_mass_table(...)["heavy"]` (sum of D+/Ds+/B+/Bc+
  production weight per POT at Ue2=1) between the two ROOT outputs.

## Findings

**1. Revised, then implemented and empirically measured: our cascade was
missing an entire particle species, confirmed by reading [64] directly.**
[64]'s cascade is a particle-stack random walk that starts with one 400
GeV/c proton and, at each step, adds **p, n, pi+/-, K+/-, KS and KL** above
the charm/beauty production threshold to the stack -- not just nucleons.
Quoting directly: *"The fraction of charm(beauty) produced with a proton as
beam particle is 65(77)%"* -- i.e. **35% of the total charm yield (23% of
beauty) comes from pion/kaon-initiated vertices**. Our `SteppingAction`
originally only watched `pdg == 2212 || pdg == 2112` (proton/neutron); it
could not produce this contribution at all, regardless of how correct the
nucleon-vertex weighting was.

Decomposing their numbers (primary = 1.0x baseline by definition, total
cascade = 2.3x charm / 1.7x beauty): proton/neutron vertices (primary +
secondary) contribute 0.65*2.3 = 1.495x charm / 0.77*1.7 = 1.309x beauty;
pion/kaon vertices contribute the remaining 0.805x charm / 0.391x beauty.
**With perfect nucleon-vertex modeling, this predicts our architecture
should be capped at 2.3/1.495 = 1.54x (charm) / 1.7/1.309 = 1.30x (beauty)
below the true cascade-enhanced yield** -- this was the theoretical
estimate before implementation.

**Implemented** (2026-10-07): `Pythia8VertexModel` now keeps one pool of
fixed-energy Pythia8 instances per *canonical* beam species --
`2212` (p/n, isospin-averaged, unchanged 74-point energy grid), `211`
(pi+/pi-, merged via exact QCD charge-conjugation invariance), `321`
(K+/K-, same). Each meson pool uses a coarser 15-point energy grid (cost
control: 2 new species x 2 sub-processes x 15 energies = 60 new Pythia8
instances) and FTFT's separate meson K-factors (`K_charm=2.02`,
`K_beauty=1.19`, vs. `2.48`/`1.04` for p/n) and pion PDF set
(`PDF:piSet=1`). `SteppingAction::IsCascadeBeamSpecies` now forwards
pi+/-/K+- cascade tracks in addition to p/n (KS/KL are not modeled -- a
documented scope limitation, not an oversight).

**Empirical validation of the species split**: a 300-proton run's
above-floor (>=35 GeV) vertex count split **62.4% nucleon- / 37.6%
meson-initiated (599/960 and 361/960)** -- close to [64]'s quoted 65%/35%
charm split, a good sign the cascade's relative meson production rate is
physically reasonable.

**Empirical measurement of the flux improvement, however, is smaller than
the theoretical estimate above.** Comparing matched 300-proton runs (same
binary, `IsCascadeBeamSpecies` restricted to nucleon-only vs. the full
p/n/pi/K set) via `flux_vs_mass_table(...)["heavy"]`
(sum of D+/Ds+/B+/Bc+ production weight per POT at Ue2=1): the ratio is
flat at **~1.11x across the full mass range** (1.112 at M_N=0.1 GeV, 1.127
at M_N=1.8 GeV, 1.113 summed over all mass bins) -- well short of the
1.3-1.54x predicted above. The vertex-count split (62/38) is close to
[64]'s, but the *flux*-weighted split is not: pion/kaon vertices that clear
the 35 GeV floor evidently carry systematically lower per-vertex weight
(softer energy spectrum within the pool's energy grid, and/or the lower
meson K-factors) than nucleon vertices at the same nominal floor. This
itself suggests [64]'s cascade (multi-generation Pythia 6.4 random walk)
and our single Geant4 FTFP_BERT cascade do not produce the same
*energy spectrum* of meson secondaries above threshold, even though the
*vertex-count* species split comes out similar -- a real, now-measured
discrepancy, not yet explained. My earlier "vertex count is not the
problem" conclusion (comparing our nucleon-only vertex count against their
nucleon+meson total enhancement factor) was not an apples-to-apples
comparison; withdrawn.

**2. `kSigmaInelasticMb = 30.0` (`Pythia8VertexModel.cc`) is the wrong
normalization, confirmed against the primary source.** Table 1 of
arXiv:1811.00930 quotes sigma_pN = 10.7 mb for exactly this role (charm
production weighting per nucleon-level vertex). Tracing its citation back
to the SHiP Technical Proposal, arXiv:1504.04956, Section 5.3, Eq. (5.15):

> "sigma_pN = 10.7 mbarn is the hadronic cross section per nucleon in a Mo
> target. The inelastic hadronic cross section per nucleon on a target
> with A nucleons can be expressed as sigma_pN = sigma_pA/A ... The
> inelastic cross section pA shows the dependence A^0.71 on the mass
> number."

This is the proton-nucleon inelastic cross section *as realized inside a
molybdenum nucleus* -- reduced from the free p-p value (~30 mb at these
energies) by nuclear shadowing (sigma_pA ~ A^0.71 is sub-linear in A, so
sigma_pA/A falls as A grows). `SteppingAction` identifies vertices at
exactly this level (one step per nucleon-level FTFP_BERT inelastic
interaction inside the Mo target), so 10.7 mb -- not the free-proton
30 mb -- is the physically consistent denominator for our vertex weight.
**Using 30 mb understates every vertex weight by 30/10.7 = 2.8x.**

**3. Our raw sigma(ccbar) is independently ~2.9x low.** The same TP
section quotes sigma_ccbar = 18.1 +/- 1.7 ubarn = 0.0181 mb directly,
which matches Table 1's X_ccbar x sigma_pN = 1.7e-3 x 10.7 mb = 0.0182 mb
almost exactly -- a useful cross-check that Table 1 is self-consistent
with this primary measurement. Our FTFT-tuned Pythia8 pool, at the
dominant 400 GeV (leading-vertex) energy, implies sigma(ccbar) ~ 0.0063 mb
once the simulated sigma(D+)+sigma(Ds+) is divided back out through Table
2's fragmentation fractions (0.207, 0.088).

**4. After fixing Finding 2 and implementing Finding 1, ~35x remains, of
which ~3.2x is explained (empirical Finding 1 x candidate Finding 3), ~12x
still open.** The original hypothesis here -- that our pool's steep energy
suppression (sigma(ccbar) falls 2-3 orders of magnitude from 400 to 35 GeV,
see table below) might not match [64]'s cascade shape -- could not be
checked quantitatively even after reading [64] directly: the note confirms
the qualitative effect (cascade hadrons have a softer momentum spectrum)
but gives no number to compare against. The empirical ~1.11x vs.
theoretical ~1.3-1.54x gap in Finding 1 above is a concrete, measured
instance of exactly this kind of shape mismatch. See "Reference [64]
methodology" below for what was actually confirmed.

## Representative pool values (K_charm = 2.48 applied)

| E_lab [GeV] | sigma(D+) [mb] | sigma(Ds+) [mb] | vertexWeight(Ds+), 30 mb norm |
|---|---|---|---|
| 400 | 3.56e-3 | 1.04e-3 | 3.48e-5 |
| 100 | 3.41e-4 | 1.01e-4 | 3.35e-6 |
| 35  | 8.49e-6 | 1.02e-6 | 3.40e-8 |

Full table available via the `Pythia8VertexModel` pool-diagnostic print
(see below).

## Reference [64] methodology (read directly, 2026-10-07)

H. Dijkstra and T. Ruf, "Heavy Flavour Cascade Production in a Beam Dump",
CERN-SHiP-NOTE-2015-009 (provided locally; not accessible via CDS/INSPIRE/
arXiv in this session, see git history of this file for that attempt).
Pythia 6.4 is used throughout, not Pythia 8.

- **Cascade algorithm**: a particle stack, seeded with one 400 GeV/c
  proton. At each step: pick a proton or neutron target (43% proton,
  matching Mo's Z/A), compare a relative probability `chi_norm` (the
  species/momentum-dependent ccbar(bbbar) cross-section ratio, normalized
  to "400 GeV/c pi+ on n") against a random number to decide whether *this*
  interaction also produces a charm/beauty event (massive matrix elements),
  then separately generate a generic inclusive QCD event for the same
  beam-target combination and add every outgoing p/n/pi+-/K+-/KS/KL above
  the charm-production kinematic threshold back onto the stack. Repeat
  until the stack empties.
- **Species breakdown** (Finding 1, revised above): 65%/35% charm,
  77%/23% beauty, nucleon-vs-meson-initiated.
- **K-factor handling differs from ours**: `chi_norm` is a *ratio* of
  Pythia cross sections, so an overall K-factor "is irrelevant, since only
  the relative cross-sections are used" -- quoting directly. The absolute
  normalization (chi = 1.7e-3/1.6e-7) comes from a separate measurement
  (SHiP TP, which chi-square-fits Pythia to data), decoupled from the
  shape/energy-dependence of the cascade random walk. We instead apply a
  single energy-independent `K_charm=2.48`/`K_beauty=1.04` directly to our
  absolute Pythia8 cross section at every pool energy -- if the true
  K-factor runs with energy (plausible this close to threshold, where
  missing higher-order corrections matter more), our shape could differ
  from theirs even after the normalization fixes above.
- **Softer cascade momentum spectrum, confirmed but not quantified**:
  "the momentum distribution of the cascade hadrons is softer than the
  hadrons produced in primary interactions" (Figs. 6-7, not digitized
  here) -- qualitatively consistent with our own pool table showing
  sigma(ccbar) falling 2-3 orders of magnitude from 400 to 35 GeV, but no
  number in the note lets this be checked quantitatively against our own
  energy-dependent suppression curve.
- Section 4's 2.6x/1.9x "HNL acceptance" (vs. 2.3x/1.7x "production")
  factors are a *different*, larger number specific to Pythia 6.4-vs-data
  pT mistuning in their setup -- not relevant to comparing our production
  rate, which should be benchmarked against the 2.3x/1.7x production
  numbers only; noted here so it isn't confused with Finding 2-4's target.

## Status and next step

Two fixes applied: `kSigmaInelasticMb = 10.7` and the pion/kaon cascade
channel (both 2026-10-07). Gap progression: 109.7x -> 39.1x -> 35.1x. Of
the original 39.1x, ~3.2x is explained (empirical ~1.11x from the now-
implemented pion/kaon channel, times the still-open ~2.9x raw cross-section
candidate from Finding 3), leaving **~12x still unexplained**. The
pion/kaon channel's smaller-than-predicted empirical effect (Finding 1) is
itself now the most concrete lead: our cascade's meson secondaries clear
the 35 GeV floor at roughly [64]'s predicted *rate* (vertex-count split
62/38 vs. their 65/35) but carry less *weight* once there (flux-ratio only
1.11x vs. 1.3-1.54x predicted) -- i.e. an energy-spectrum mismatch, not a
vertex-count mismatch. Remaining candidates, in rough order of suspicion:
(a) that energy-spectrum mismatch itself -- FTFP_BERT's meson secondaries
vs. [64]'s Pythia-6.4-cascade meson secondaries may carry different
momentum distributions above the shared 35 GeV floor, (b) the K-factor's
energy-dependence (we apply a single value per species at every pool
energy; [64]'s ratio-based `chi_norm` sidesteps this entirely), (c) our
FTFT tune (fit to pp/pi-N fixed-target *production* data) vs. [64]'s
separate E791-based acceptance-shape tune -- not necessarily consistent
with each other, (d) possible differences in how nuclear effects (nothing
beyond isospin-averaging is modeled in our `Pythia8VertexModel`, see its
header) enter each calculation. None of (a)-(d) have been checked
quantitatively yet.

## Diagnostic instrumentation left in the code

- `SteppingAction.cc`: vertex-floor counters (total vertices, vertices
  >= 35 GeV, split by nucleon- vs. meson-initiated), printed in the
  destructor.
- `Pythia8VertexModel.cc`: per-pool-energy cross-section/vertex-weight
  table, printed per beam species at the end of the constructor.

Both commented out (2026-10-07) now that this investigation's numbers are
recorded above -- uncomment to re-enable for future runs.
