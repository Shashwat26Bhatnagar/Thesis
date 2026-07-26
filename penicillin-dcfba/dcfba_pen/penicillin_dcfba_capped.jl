#=
  penicillin_dcfba.jl  --  Penicillin fed-batch optimisation with dcFBA.

  Adapted from Gotsmy et al. (2024) dcFBA  T_v2024.jl
  (https://github.com/Gotsmy/dcFBA).  The single-level KKT reformulation,
  orthogonal collocation, moving finite elements and IPOPT solve are UNCHANGED.

  WHAT CHANGED vs the original (4 edits, all flagged with  ### PEN):
   1. State vector N = [glucose, biomass, penicillin]  (sulfate removed).
   2. Production envelope get_q_P(...)  ->  Bajpai-Reuss  c(S,X) = qp(S)/mu(S,X);
      product constraint  q[pro] = q_P  ->  q[pro] = c * q[xxx]   (pi = c*mu).
   3. Glucose uptake pinned to the kinetic Monod rate vg(S) (NOT to the feed),
      so glucose S evolves and c(S,X) varies over the run.
   4. Objective: maximise final penicillin titer  P(T)/V(T)  (eq 2a).

  To use the genome-scale model iAL1006 instead of the core:
   - write its S/LB/UB to 00_files/ and set glu/xxx/pro/atp to its reaction
     indices; nothing else changes.

  Run:  julia penicillin_dcfba.jl     (needs JuMP, Ipopt[+ma27], DelimitedFiles,
                                        JLD2, DataFrames, CSV)
=#
using JuMP
using LinearAlgebra
using Ipopt
using DelimitedFiles
using JLD2
import DataFrames
import CSV

include("00_files/postprocessing_02.jl")   # reuse dcFBA helpers (get_idx_lists, ...)

# ----------------------------------------------------------------------------- #
### PEN  OUTER process objective (what the feed profile is optimised for):
#   "total_pen" -> max P(T)        total penicillin produced   [g]   (default)
#   "titer"     -> max P(T)/V(T)   end-of-feed concentration   [g/L] (eq 2a)
#   "biomass"   -> max X(T)        biomass only (NOT recommended: high glucose
#                                  represses penicillin -> cells but little drug)
const OBJECTIVE = "total_pen"
# ----------------------------------------------------------------------------- #
### PEN  Bajpai-Reuss (1980) M4 parameters, O2 non-limiting (Kox=Kop=0)
const MU_X = 0.092     # max specific growth rate   [1/h]
const K_X  = 0.15      # Contois constant           [g/g]
const MU_P = 0.005     # max specific prod. rate     [1/h]   (rescale to strain)
const K_P  = 2.0e-4    # Monod constant, product     [g/L]
const K_I  = 0.1       # substrate inhibition        [g/L]
# glucose uptake kinetics
const VGMAX = 0.35     # g glc / gDW / h
const KG    = 0.5      # g/L
const KIUP  = 200.0    # g/L
### PEN  molecular weights for unit conversion (GEM fluxes are in mmol, BR in g)
const MW_GLC  = 0.18016   # g/mmol glucose
const MW_PENG = 0.33439   # g/mmol penicillin-G (C16H18N2O4S)
const S_CAP   = 40.0      # PEN: max substrate concentration [g/L] (osmotic/inhibition cap)

### PEN  EXPERIMENTAL time-varying penicillin/biomass ratio  c(t) = P(t)/X(t)
### [g penicillin / g biomass], ONE value per finite element (i = 1..nFE).
### Generated from measured X(t),P(t) by compute_ct.py -- REPLACE with your data.
### Source: IndPenSim Batch 29 (highest-penicillin batch, 36.18 g/L), c=P/X per FE.
const C_PX = [0.0007, 0.0222, 0.2255, 0.6403, 1.0178, 1.2504, 1.3700, 1.4268, 1.4527, 1.4643]

mu_BR(S,X) = MU_X * S / (K_X * X + S + 1e-9)                 # Contois (Eq 16)
qp_BR(S)   = MU_P * S / (K_P + S * (1.0 + S / K_I) + 1e-12)  # Haldane (Eq 17)
c_BR(S,X)  = qp_BR(S) / (mu_BR(S,X) + 1e-9)                  # (unused now; kept for reference)
vg_BR(S)   = VGMAX * S / (KG + S + S*S/KIUP)                 # kinetic uptake limit
# ----------------------------------------------------------------------------- #


function fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,xxx,pro,vlb,vub,d)
    A_idx, B_idx, C_idx, D_idx = get_idx_lists(vlb,vub)
    nA=size(A_idx,1); nB=size(B_idx,1); nC=size(C_idx,1); nD=size(D_idx,1)
    println("nA=$nA nB=$nB nC=$nC nD=$nD")

    nR = size(S,2); nM = size(S,1); nN = length(N0); nCP = 3
    println("nr of KKT variables = ", nFE*(2*nA+nB+nC+nM))

    w=1e-6; phi1=1e2; phi2=1e1   # w = pFBA regularisation (was 1e-20: too small)
    colmat = [0.19681547722366 -0.06553542585020 0.02377097434822;
              0.39442431473909  0.29207341166523 -0.04154875212600;
              0.37640306270047  0.51248582618842 0.11111111111111]
    radau  = [0.15505 0.64495 1.00000]

    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
            "warm_start_init_point"=>"yes", "print_level"=>5,
            "linear_solver"=>"mumps",            # bundled, no HSL licence needed
            "max_iter"=>Int(1e5),
            "tol"=>1e-3, "acceptable_iter"=>30, "acceptable_tol"=>1e-1,
            "nlp_scaling_method"=>"gradient-based",  # auto-scale ill-conditioned problem
            "mu_strategy"=>"adaptive",
            "bound_relax_factor"=>1e-8,
            "bound_push"=>1e-6, "max_soc"=>4,
            "watchdog_shortened_iter_trigger"=>10))

    @variables(m, begin
        N[1:nN,1:nFE,1:nCP]; Ndot[1:nN,1:nFE,1:nCP]
        V[1:nFE,1:nCP];      Vdot[1:nFE,1:nCP]
        q[1:nR,1:nFE]
        lambda_Sv[1:nM,1:nFE]
        alpha_ub_A[1:nA,1:nFE]; slack_ub_A[1:nA,1:nFE]
        alpha_lb_A[1:nA,1:nFE]; slack_lb_A[1:nA,1:nFE]
        alpha_lb_B[1:nB,1:nFE]; slack_lb_B[1:nB,1:nFE]
        alpha_ub_C[1:nC,1:nFE]; slack_ub_C[1:nC,1:nFE]
        t_end; rtFE[1:nFE]; F[1:nFE]
    end)

    # start values
    for i in 1:nFE, j in 1:nCP
        for k in 1:nN; set_start_value(N[k,i,j], N0[k]); end
        set_start_value(V[i,j], V0)
    end
    for i in 1:nFE; set_start_value(rtFE[i], 1); end

    ### PEN  state indices:  N[1]=glucose [mmol], N[2]=biomass [g], N[3]=penicillin [mmol]
    @NLexpressions(m, begin
        tFE[i=1:nFE], t_end/nFE*rtFE[i]
        # glucose concentration in g/L for Bajpai-Reuss  (mmol/L * g/mmol)
        S_gL[i=1:nFE], (N[1,i,2]/V[i,2]) * MW_GLC
        # kinetic glucose uptake [mmol/(gDW h)] = vg_BR[g/(gDW h)] / MW_GLC
        q_G[i=1:nFE], vg_BR(S_gL[i]) / MW_GLC
        # c(S,X) = qp/mu  [g/g]   (S in g/L, X = biomass conc g/L)
        c_t[i=1:nFE], C_PX[i]                                 # PEN: time-varying P/X ratio per FE
    end)

    ### PEN  outer objective selectable via OBJECTIVE flag (see top of file)
    @NLexpression(m, slack_pen,
        sum(
            sum(-slack_lb_A[mc,i] for mc in 1:nA) +
            sum(-slack_ub_A[mc,i] for mc in 1:nA) +
            sum(-slack_lb_B[mc,i] for mc in 1:nB) +
            sum(-slack_ub_C[mc,i] for mc in 1:nC)
        for i in 1:nFE)/nFE)

    if OBJECTIVE == "titer"
        @NLobjective(m, Max, (N[3,end,end]/V[end,end])*phi1 - slack_pen)   # P/V
    elseif OBJECTIVE == "biomass"
        @NLobjective(m, Max,  N[2,end,end]*phi1 - slack_pen)               # X(T)
    else  # "total_pen"
        @NLobjective(m, Max,  N[3,end,end]*phi1 - slack_pen)               # P(T)
    end

    @NLconstraints(m, begin
        # glucose uptake pinned to kinetic rate; EX_GLCb is negative for uptake
        NLc_q_G[i=1:nFE],  q[glu,i] + q_G[i] == 0
        ### PEN  penicillin coupling:  pi = c*mu, converted to mmol/(gDW h)
        #   q[pro] [mmol/gDW/h] = c [g/g] * mu [1/h] / MW_PENG [g/mmol]
        NLc_q_P[i=1:nFE],  q[pro,i] - c_t[i]*q[xxx,i]/MW_PENG == 0
        NLc_t_end, sum(tFE[i] for i in 1:nFE) == t_end
        # KKT complementary slackness
        NLc_slack_lb_A[mc=1:nA,i=1:nFE], slack_lb_A[mc,i] == (q[A_idx[mc],i]-vlb[A_idx[mc]])*alpha_lb_A[mc,i]/nA*phi2
        NLc_slack_ub_A[mc=1:nA,i=1:nFE], slack_ub_A[mc,i] == (q[A_idx[mc],i]-vub[A_idx[mc]])*alpha_ub_A[mc,i]/nA*phi2
        NLc_slack_lb_B[mc=1:nB,i=1:nFE], slack_lb_B[mc,i] == (q[B_idx[mc],i]-vlb[B_idx[mc]])*alpha_lb_B[mc,i]/nB*phi2
        NLc_slack_ub_C[mc=1:nC,i=1:nFE], slack_ub_C[mc,i] == (q[C_idx[mc],i]-vub[C_idx[mc]])*alpha_ub_C[mc,i]/nC*phi2
        # collocation (2nd..nth point)
        coll_N[l=1:nN,i=2:nFE,j=1:nCP], N[l,i,j]==N[l,i-1,nCP]+tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
        coll_V[       i=2:nFE,j=1:nCP], V[i,j]  ==V[  i-1,nCP]+tFE[i]*sum(colmat[j,k]*Vdot[i,k]   for k in 1:nCP)
        # collocation (1st point)
        coll_N0[l=1:nN,i=[1],j=1:nCP], N[l,i,j]==N0[l]+tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
        coll_V0[       i=[1],j=1:nCP], V[i,j]  ==V0  +tFE[i]*sum(colmat[j,k]*Vdot[i,k]   for k in 1:nCP)
    end)

    # bounds
    for mc in 1:nR, i in 1:nFE
        set_lower_bound(q[mc,i],vlb[mc]); set_upper_bound(q[mc,i],vub[mc])
        for j in 1:nCP
            for n in 1:nN; set_lower_bound(N[n,i,j],0); end
            set_lower_bound(V[i,j],V0); set_upper_bound(V[i,j],V_max)
        end
    end
    for i in 1:nFE
        set_lower_bound(rtFE[i],0.8); set_upper_bound(rtFE[i],1.2)
        set_lower_bound(F[i],0.0);    set_upper_bound(F[i],0.05)   # feed bound [L/h]
        for mc in 1:nA; set_upper_bound(alpha_lb_A[mc,i],0); set_lower_bound(alpha_ub_A[mc,i],0); end
        for mc in 1:nB; set_upper_bound(alpha_lb_B[mc,i],0); end
        for mc in 1:nC; set_lower_bound(alpha_ub_C[mc,i],0); end
    end
    set_lower_bound(t_end,t_min); set_upper_bound(t_end,t_max)

    @constraints(m, begin
        ### PEN  differential equations (eqs 1a-1d), total amounts
        # glucose [mmol]:  dG = F*c_G + q[glu]*X   (q[glu] negative = uptake)
        m1[i=1:nFE,j=1:nCP], Ndot[1,i,j] == F[i]*c_G + q[glu,i]*N[2,i,j]
        # biomass:   dX = mu*X
        m2[i=1:nFE,j=1:nCP], Ndot[2,i,j] == q[xxx,i]*N[2,i,j]
        # penicillin:dP = pi*X
        m3[i=1:nFE,j=1:nCP], Ndot[3,i,j] == q[pro,i]*N[2,i,j]
        # volume:    dV = F
        v1[i=1:nFE,j=1:nCP], Vdot[i,j]   == F[i]
        # PEN: cap substrate CONCENTRATION  S = G/V*MW_GLC <= S_CAP  (linear form)
        Scap[i=1:nFE,j=1:nCP], MW_GLC*N[1,i,j] <= S_CAP*V[i,j]
        # steady-state metabolism  S v = 0
        c_S[mc=1:nM,i=1:nFE], sum(S[mc,k]*q[k,i] for k in 1:nR) == 0
        # KKT stationarity  (grad_v L = 0)
        Lagr_A[mc=1:nA,i=1:nFE], d[A_idx[mc]]+w*q[A_idx[mc],i]+alpha_lb_A[mc,i]+alpha_ub_A[mc,i]+sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM)==0
        Lagr_B[mc=1:nB,i=1:nFE], d[B_idx[mc]]+w*q[B_idx[mc],i]+alpha_lb_B[mc,i]+sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM)==0
        Lagr_C[mc=1:nC,i=1:nFE], d[C_idx[mc]]+w*q[C_idx[mc],i]+alpha_ub_C[mc,i]+sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM)==0
        Lagr_D[mc=1:nD,i=1:nFE], d[D_idx[mc]]+w*q[D_idx[mc],i]+sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM)==0
        c_alpha_lb_A[mc=1:nA,i=1:nFE], alpha_lb_A[mc,i]<=0
        c_alpha_ub_A[mc=1:nA,i=1:nFE], alpha_ub_A[mc,i]>=0
        c_alpha_lb_B[mc=1:nB,i=1:nFE], alpha_lb_B[mc,i]<=0
        c_alpha_ub_C[mc=1:nC,i=1:nFE], alpha_ub_C[mc,i]>=0
    end)

    println("Preprocessing finished. Optimising ...")
    status = JuMP.optimize!(m); t = solve_time(m)
    N_=JuMP.value.(N); Ndot_=JuMP.value.(Ndot); V_=JuMP.value.(V); Vdot_=JuMP.value.(Vdot)
    q_=JuMP.value.(q); tFE_=JuMP.value.(tFE); F_=JuMP.value.(F); t_end_=JuMP.value.(t_end)
    @JLD2.save dirname*"/variables.jld2" N_ Ndot_ V_ Vdot_ q_ tFE_ F_ t_end_
    return N_, Ndot_, V_, Vdot_, q_, tFE_, m, t
end


function run_kkt_simulation()
    S   = readdlm("00_files/S.csv")
    vlb = readdlm("00_files/LB.csv")[:,1]
    vub = readdlm("00_files/UB.csv")[:,1]
    println("nR = ", size(vlb,1))

    ### PEN  REAL iAL1006 PARSIMONIOUS CORE (565 rxns) indices (index_map.json):
    glu = 2      # EX_GLCb   (negative = glucose uptake)
    xxx = 532    # r1348     (growth, flux = mu [1/h])
    pro = 3      # EX_PENGb  (penicillin-G secretion)
    atp = 553    # r1466     (ATP maintenance)
    vlb[glu]=-100; vub[glu]=0          # uptake pinned by NLc_q_G to vg_BR(S)
    vlb[xxx]=0;    vub[xxx]=1000
    vlb[pro]=0;    vub[pro]=1000

    # initial state  N0 = [glucose mmol, biomass g, penicillin mmol]
    V0=0.5; S0_gL=15.0; G0=S0_gL/MW_GLC*V0; X0=0.5; N0=[G0, X0, 0.0]
    c_G = 500.0 / MW_GLC               # feed glucose 500 g/L -> mmol/L
    t_max=150.0; t_min=150.0; V_max=2.0; nFE=10
    @assert length(C_PX) == nFE "C_PX must have one entry per finite element (length == nFE). Re-run compute_ct.py with N_FE=$nFE."

    nR=size(S,2); d=zeros(nR); d[xxx]=-1.0   # inner FBA: max biomass (zeros, NOT undef)

    N_,Ndot_,V_,Vdot_,q_,tFE_,m_,t_ =
        fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,xxx,pro,vlb,vub,d)
    println("DONE")

    summary = "Termination: "*string(termination_status(m_))*
              "\nTime [s]: "*string(t_)*
              "\nObjective ("*OBJECTIVE*"): "*string(JuMP.objective_value(m_))*
              "\nFinal penicillin [g]: "*string(N_[3,end,end]*MW_PENG)*
              "\nFinal biomass [g]: "*string(N_[2,end,end])*
              "\nFinal volume [L]: "*string(V_[end,end])*
              "\nProcess end [h]: "*string(round(sum(tFE_),digits=2))*
              "\nFeed rates [L/h]: "*string([round(i,digits=4) for i in Vdot_[:,1]])*
              "\nGrowth rates [1/h]: "*string([round(i,digits=3) for i in q_[xxx,:]])
    println(summary); write(dirname*"/summary.txt", summary)

    df = DataFrames.DataFrame(hcat(
            get_time_points(tFE_), get_points(V_,tFE_,V0),
            get_points(N_,tFE_,N0), get_points(Vdot_,tFE_),
            get_points(Ndot_,tFE_), get_fluxes(q_,[glu,xxx,pro,atp],tFE_)),
        ["t","V","G","X","P","r_V","r_G","r_X","r_P","q_G","q_X","q_P","q_M"])
    writedlm(dirname*"/q.csv", q_, ','); CSV.write(dirname*"/df.csv", df)
end

println("Start"); dirname = PROGRAM_FILE[begin:end-3]; mkpath(dirname)
run_kkt_simulation()
println("End")
