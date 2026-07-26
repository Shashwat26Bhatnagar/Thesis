# Penicillin fed-batch optimisation: dcFBA + Bajpai-Reuss c(t) on **real iAL1006**

Adapts the **dcFBA** framework of Gotsmy et al. (2024) for **penicillin**, with the
time-varying penicillin/biomass ratio **c(t)=qp/μ** from **Bajpai & Reuss (1980)**,
running on the **real genome-scale model iAL1006** (Ågren et al. 2013) — not a
hand-built toy.

## The model is real now

`iAL1006.xml` is the published *P. chrysogenum* Wisconsin54-1255 GEM, taken from the
**RAVEN Toolbox** repo (`SysBioChalmers/RAVEN/tutorial/iAL1006 v1.00.xml`),
1792 reactions / 1395 metabolites / 1006 ORFs.

The CSVs are extracted directly from it under a **defined glucose medium**
(glucose-limited + O₂ + NH₃ + sulfate + phosphate + thiamine + trace cofactor + PAA),
which gives realistic glucose-limited growth (μ = 0.085 h⁻¹ at 1 mmol glc/gDW/h,
matching Bajpai-Reuss μ_x = 0.092):

```
dcfba_pen/00_files/   condensed model used by the dcFBA   1042 mets × 1292 rxns
ial1006_full/         the full model + original SBML       1395 mets × 1792 rxns
```

Condensation keeps the union of growth-active and penicillin-active reactions (so the
ACV→IPN→Pen-G pathway is retained), exactly as Gotsmy condensed iML1515.

Reaction indices (1-based, condensed) — in `00_files/index_map.json`:

| role | reaction | index |
|---|---|---|
| glucose uptake | EX_GLCb (neg = uptake) | 6 |
| O₂ | EX_O2b | 23 |
| biomass / growth (μ) | r1348 | 1226 |
| penicillin-G (π) | EX_PENGb | 10 |
| ATP maintenance | r1466 | 1260 |

## What the adaptation changes vs the original `T_v2024.jl` (flagged `### PEN`)

1. **State** `N = [glucose mmol, biomass g, penicillin mmol]`.
2. **Production**: Bajpai-Reuss `c(S,X)=qp/μ` replaces their pDNA envelope; constraint
   `q[pro] = c·q[xxx]/MW_PENG` (π = c·μ, with mmol↔g unit conversion).
3. **Glucose uptake** pinned to kinetic `vg_BR(S)` (EX_GLCb is negative for uptake), so
   glucose **S evolves** and `c(S,X)` varies.
4. **Objective** selectable (`OBJECTIVE` flag): `total_pen` (default) | `titer` | `biomass`.

Unit conversions (GEM is in mmol, Bajpai-Reuss in g): `MW_GLC=0.18016`, `MW_PENG=0.33439` g/mmol.

## Verified on the real model (Python cross-check)

`verify_ial1006.py` runs the explicit dFBA on the **real condensed iAL1006 CSVs**
(`00_files/`) with the c(t) coupling and the unit conversions. It produces
`verify_ial1006.png`: glucose batch phase → glucose depletion → penicillin switches on,
**titer ≈ 15.6 g/L** — a realistic penicillin fed-batch. Run it to confirm the matrices:

```bash
cd dcfba_pen
pip install cobra scipy numpy matplotlib
python3 verify_ial1006.py
```

## Run the Julia dcFBA (the optimiser)

```bash
cd dcfba_pen
# Julia pkgs: JuMP, Ipopt (+HSL ma27; or switch to "mumps"), DelimitedFiles, JLD2, DataFrames, CSV
julia penicillin_dcfba.jl
```
Writes `penicillin_dcfba/summary.txt`, `df.csv`, `q.csv`, `variables.jld2`
(optimal feed profile, trajectories, fluxes).

## Files

```
dcfba_pen/
  penicillin_dcfba.jl          adapted dcFBA, wired to real iAL1006 + indices + units
  verify_ial1006.py            runnable Python cross-check on the real CSVs
  verify_ial1006.png           result: real iAL1006 fed-batch, titer ~15.6 g/L
  validation_ct_dynamics.png   earlier core-model c(t) switch figure
  00_files/
    S.csv, LB.csv, UB.csv      REAL condensed iAL1006 (1042 × 1292), defined medium
    reactions.txt              index -> reaction id/name
    index_map.json             glu/o2/bio/penG/atpm indices
    postprocessing_02.jl       dcFBA helpers (from upstream repo)
ial1006_full/
    iAL1006.xml                original published SBML (RAVEN repo)
    S.csv, LB.csv, UB.csv      REAL full model (1395 × 1792), defined medium
    reactions.txt
prototype/                     toy-core Python dFBA (kept for reference/teaching)
```

## Honest caveats

- **Julia not executed here** (no JuMP/IPOPT in this sandbox). The model wiring, indices,
  units and the c(t) coupling are verified in Python on the real CSVs; review/run the
  Julia in your environment.
- The condensed model is **1292 reactions** — the dcFBA KKT-NLP is heavy at that size.
  If IPOPT struggles, reduce `nFE`, tighten the medium to shrink the model further, or
  start from a good warm start.
- Pen-G is synthesised here without external PAA (the model makes the side chain
  internally); feed PAA (EX_PAAb) if you want precursor-limited behaviour. For Pen-V use
  EX_PENVb + POA (not active in this medium).
- `π = c·μ` re-couples π to μ once `c` hits its cap at low glucose; for strict decoupling
  constrain `q[pro] = qp_BR(S)/MW_PENG` instead (one line in `NLc_q_P`).

## Sources
- iAL1006: Ågren et al. 2013, PLoS Comput Biol 9(3):e1002980 (model via SysBioChalmers/RAVEN).
- dcFBA: Gotsmy et al. 2024, bioRxiv 2024.06.11.598442 / IFAC-PapersOnLine.
- Bajpai & Reuss 1980, J Chem Tech Biotechnol 30:332-344.
