import numpy as np, networkx as nx
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import model_learning.pensim_data as pdata
from RVGP.geometry import manifold_graph

N_KEEP, N_NEIGHBORS, FRAC = 300, 10, 1.5
obs, act, nobs = pdata.load_offline(normalize=True, max_transitions=4000)
obs_s, act_s, nobs_s = pdata.subsample(obs, act, nobs, n_keep=N_KEEP)
Z = np.hstack([obs_s, act_s]).astype(np.float64)
N, p = Z.shape
nn = int(N_NEIGHBORS*FRAC)
print(f"Z {Z.shape}  geodesic nn={nn}  required d={p}")

tree = cKDTree(Z)
pairs = tree.query_pairs(r=1e-10)
print(f"\nexact duplicate pairs: {len(pairs)}", list(pairs)[:10])
d1,_ = tree.query(Z, k=2)
print(f"NN dist min={d1[:,1].min():.3e} median={np.median(d1[:,1]):.3e}")
print(f"points with NN dist < 1e-8: {(d1[:,1]<1e-8).sum()}")

G = manifold_graph(Z, typ="knn", n_neighbors=N_NEIGHBORS)
A = nx.to_scipy_sparse_array(G, weight=None, format="csr").astype(float)
print(f"\ngraph {G.number_of_nodes()}n {G.number_of_edges()}e connected={nx.is_connected(G)}")

D = dijkstra(csr_matrix(A), directed=False)
fail=[]; ranks=np.zeros(N,int)
for i in range(N):
    o = np.argsort(D[i]); o = o[np.isfinite(D[i][o])][:nn+1]
    idx = o[o!=i][:nn]
    E = Z[idx]-Z[i]
    if len(idx) < p:
        fail.append((i,"too_few",len(idx),np.nan)); ranks[i]=-1; continue
    sv = np.linalg.svd(E.T, compute_uv=False)
    f = np.zeros(p); f[:len(sv)]=sv
    r = int((f > 1e-8*max(f[0],1e-30)).sum()); ranks[i]=r
    if r < p: fail.append((i,"rank_deficient",r,f[p-1]))

print(f"\n=== {len(fail)}/{N} points FAIL (need rank {p}) ===")
ok = ranks[ranks>=0]
print(f"local rank min={ok.min()} median={int(np.median(ok))} max={ok.max()}")
for i,w,r,sv in fail[:20]: print(f"{i:5d} {w:>16} rank={r:3d} smallest_sv={sv:.3e}")

if fail:
    b = np.array([f[0] for f in fail])
    print(f"\nfailing idx range {b.min()}..{b.max()}  count={len(b)}")
    i = b[0]
    o = np.argsort(D[i]); o=o[np.isfinite(D[i][o])][:nn+1]; idx=o[o!=i][:nn]
    sv = np.linalg.svd((Z[idx]-Z[i]).T, compute_uv=False)
    print(f"\npoint {i} singular values:", np.round(sv,8))
    print("neighbourhood per-dim std:", np.round(Z[idx].std(0),8))
    print("CONSTANT dims in this neighbourhood:", np.where(Z[idx].std(0)<1e-10)[0])
