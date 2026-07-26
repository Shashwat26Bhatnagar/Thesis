import numpy as np, model_learning.pensim_data as pdata
obs, act, nobs, so, sa = pdata.load_offline(max_transitions=4000)
o,a,n = pdata.subsample(obs, act, nobs, n_keep=300)
Z = np.hstack([o,a])
pdata.local_rank_report(Z, Ks=(15,20,30,40,60))
pdata.flat_channel_report(Z, K=40)
