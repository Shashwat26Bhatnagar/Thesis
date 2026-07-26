import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
import plotly.graph_objects as go
import plotly.express as px

MW_GLC,MW_PENG=0.18016,0.33439
cG=500.0/MW_GLC; F_SNAP,C_SNAP=0.05,0.23
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
_vg=lambda S:np.interp(S,Sg,vg_t)
S=np.loadtxt("00_files/S.csv"); LB=np.loadtxt("00_files/LB.csv"); UB=np.loadtxt("00_files/UB.csv")
nM,nR=S.shape; Ssp=sp.csr_matrix(S); glu,xxx,pro=1,531,2
def FBA(vg_ub,c):
    b=list(zip(LB,UB)); b[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR)); row[0,pro]=1.0; row[0,xxx]=-c/MW_PENG
    A=sp.vstack([Ssp,row]).tocsr()
    r=linprog(np.eye(1,nR,xxx).ravel()*-1,A_eq=A,b_eq=np.zeros(nM+1),bounds=b,method="highs")
    return (r.x[xxx],r.x[glu]) if r.success else (0.0,0.0)

roll=np.load("denoiser_rollout.npz")["roll"]
nstep=roll.shape[0]-1; npart=roll.shape[1]
Gr,Xr,Vr=roll[:,:,0],roll[:,:,1],roll[:,:,2]

gG=np.linspace(50,1700,6); gX=np.linspace(0.5,10,6); gV=np.array([0.6,0.87,1.15])
print("computing field ...")
rows=[]
for G in gG:
    for X in gX:
        for V in gV:
            S_gL=max(G,0.)/max(V,1e-6)*MW_GLC
            mu,vglu=FBA(_vg(S_gL)/MW_GLC,C_SNAP)
            dG=F_SNAP*cG+vglu*X; dX=mu*X; dV=F_SNAP
            mag=np.sqrt(dG**2+dX**2+dV**2)+1e-12; sc=55.0
            rows.append([G,X,V,dG/mag*sc,dX/mag*sc,dV/mag*sc,mag])
arr=np.array(rows); spd=(arr[:,6]-arr[:,6].min())/(arr[:,6].max()-arr[:,6].min())

traces=[]

# field cones
traces.append(go.Cone(
    x=arr[:,0],y=arr[:,1],z=arr[:,2],
    u=arr[:,3],v=arr[:,4],w=arr[:,5],
    colorscale="Viridis",cmin=0,cmax=1,
    sizemode="absolute",sizeref=28,
    showscale=True,
    colorbar=dict(title=dict(text="field speed",font=dict(color="white")),
                  tickfont=dict(color="white"),x=1.02,len=0.45,y=0.75),
    hovertemplate="G=%{x:.0f} X=%{y:.2f} V=%{z:.3f}<extra>field</extra>",
    name="vector field",anchor="tail",opacity=0.8,
))

# trajectories coloured by step
colors=px.colors.sample_colorscale("Plasma",nstep)
for j in range(0,npart,2):
    for k in range(nstep):
        traces.append(go.Scatter3d(
            x=[Gr[k,j],Gr[k+1,j]],y=[Xr[k,j],Xr[k+1,j]],z=[Vr[k,j],Vr[k+1,j]],
            mode="lines",line=dict(color=colors[k],width=2),
            showlegend=(j==0 and k==0),
            name="forward trajectories" if (j==0 and k==0) else "",
            hoverinfo="skip",opacity=0.7,
        ))

# initial cloud
traces.append(go.Scatter3d(
    x=Gr[0,:],y=Xr[0,:],z=Vr[0,:],mode="markers",
    marker=dict(size=4,color="royalblue",opacity=0.9),
    name="p(s,0)  initial",
    hovertemplate="G=%{x:.1f} X=%{y:.3f} V=%{z:.4f}<extra>initial</extra>",
))

# final cloud
traces.append(go.Scatter3d(
    x=Gr[-1,:],y=Xr[-1,:],z=Vr[-1,:],mode="markers",
    marker=dict(size=7,color="crimson",symbol="diamond",opacity=0.95),
    name="p(s,T)  denoised",
    hovertemplate="G=%{x:.1f} X=%{y:.3f} V=%{z:.4f}<extra>denoised</extra>",
))

# dummy trace for trajectory step colorbar
traces.append(go.Scatter3d(
    x=[None],y=[None],z=[None],mode="markers",
    marker=dict(size=0,color=[0,nstep],colorscale="Plasma",
                colorbar=dict(title=dict(text="traj step",font=dict(color="white")),
                              tickfont=dict(color="white"),x=1.12,len=0.45,y=0.25),
                showscale=True),
    showlegend=False,hoverinfo="skip",
))

ax_style=dict(backgroundcolor="#1a1a1a",gridcolor="#333",
              zerolinecolor="#555",tickfont=dict(color="white"))

fig=go.Figure(data=traces)
fig.update_layout(
    title=dict(
        text="FL-GFN forward denoiser on dcFBA vector field — drag to rotate · scroll to zoom · hover for values",
        font=dict(size=14,color="white"),x=0.5),
    paper_bgcolor="#111",
    scene=dict(
        xaxis=dict(title=dict(text="G  glucose [mmol]",font=dict(color="white")),**ax_style),
        yaxis=dict(title=dict(text="X  biomass [g]",font=dict(color="white")),**ax_style),
        zaxis=dict(title=dict(text="V  volume [L]",font=dict(color="white")),**ax_style),
        bgcolor="#111",
        camera=dict(eye=dict(x=1.6,y=-1.6,z=0.8)),
    ),
    legend=dict(font=dict(color="white"),bgcolor="#222",bordercolor="#444"),
    margin=dict(l=0,r=140,t=60,b=0),
    width=1100,height=750,
)

out="denoiser_field_interactive.html"
fig.write_html(out,include_plotlyjs="cdn")
import os; print(f"saved {out}  ({os.path.getsize(out)//1024} KB)")
