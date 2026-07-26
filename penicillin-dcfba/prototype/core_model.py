"""
Compact P. chrysogenum penicillin core model (cobrapy) for the dFBA demo.

This is a minimal, carbon-consistent stoichiometric model (mass basis, g/gDW/h),
calibrated to Bajpai-Reuss yields, so the FBA layer is a REAL cobra LP in which
penicillin competes with biomass for carbon. The objective is biomass.

  GLC_up :  glc_e -> glc         (glucose uptake, bounded by kinetics)
  BIO    :  (1/Yxs) glc -> bio   (growth; objective)  -> flux = mu  [1/h]
  PEN    :  (1/Yps) glc -> pen   (penicillin)         -> flux = pi  [g/gDW/h]
  ATPM   :  mx glc ->            (maintenance carbon drain)
  EX_bio, EX_pen sinks.

>>> TO USE THE GENOME-SCALE MODEL iAL1006 INSTEAD:
    replace build_core_model() with:
        model = cobra.io.read_sbml_model("iAL1006.xml")
    and set RXN ids below to iAL1006's biomass / penicillin / glucose reactions.
    Everything downstream (FBA_pen, dfba, optimiser) is unchanged.
"""
from __future__ import annotations
import cobra
from cobra import Model, Reaction, Metabolite

# Bajpai-Reuss yields
YXS = 0.45   # g DW / g glucose
YPS = 0.9    # g penicillin / g glucose
MX  = 0.014  # g glucose / gDW / h  (maintenance)

# reaction ids the rest of the code references (model-agnostic indirection)
RXN = dict(bio="BIO", pen="PEN", glc="GLC_up")


def build_core_model():
    m = Model("Pchr_core")
    glc_e = Metabolite("glc_e", name="glucose (ext)")
    glc   = Metabolite("glc_c", name="glucose (cyt)")
    bio   = Metabolite("biomass_c", name="biomass")
    pen   = Metabolite("pen_c", name="penicillin")

    GLC_up = Reaction("GLC_up"); GLC_up.add_metabolites({glc_e: -1, glc: 1})
    GLC_up.lower_bound, GLC_up.upper_bound = 0.0, 1000.0
    EX_glc = Reaction("EX_glc_e"); EX_glc.add_metabolites({glc_e: -1})
    EX_glc.lower_bound, EX_glc.upper_bound = -1000.0, 0.0   # supply

    BIO = Reaction("BIO"); BIO.add_metabolites({glc: -1.0 / YXS, bio: 1.0})
    BIO.lower_bound, BIO.upper_bound = 0.0, 1000.0
    PEN = Reaction("PEN"); PEN.add_metabolites({glc: -1.0 / YPS, pen: 1.0})
    PEN.lower_bound, PEN.upper_bound = 0.0, 1000.0
    ATPM = Reaction("ATPM"); ATPM.add_metabolites({glc: -1.0})
    ATPM.lower_bound, ATPM.upper_bound = 0.0, 1000.0   # set per-step to mx*... if desired

    EX_bio = Reaction("EX_bio"); EX_bio.add_metabolites({bio: -1}); EX_bio.bounds = (0, 1000)
    EX_pen = Reaction("EX_pen"); EX_pen.add_metabolites({pen: -1}); EX_pen.bounds = (0, 1000)

    m.add_reactions([GLC_up, EX_glc, BIO, PEN, ATPM, EX_bio, EX_pen])
    m.objective = "BIO"
    return m


if __name__ == "__main__":
    m = build_core_model()
    m.reactions.GLC_up.upper_bound = 1.0   # 1 g glc/gDW/h available
    s = m.optimize()
    print("max-biomass-only:  mu =", round(s.fluxes["BIO"], 4),
          " pen =", round(s.fluxes["PEN"], 4))
