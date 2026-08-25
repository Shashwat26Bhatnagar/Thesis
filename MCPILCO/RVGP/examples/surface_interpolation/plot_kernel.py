#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import RVGP
from RVGP.utils import load_mesh
from RVGP.geometry import furthest_point_sampling
from RVGP.kernels import ManifoldKernel
import polyscope as ps
import numpy as np
import gpflow

n_eigenpairs=100
vertices, faces = load_mesh('sphere')

sample_ind, _ = furthest_point_sampling(vertices, spacing=0.05)
X = vertices[sample_ind]

train_ind =  np.random.choice(np.arange(len(X)), size=int(0.5*len(X)))
test_ind = [i for i in range(len(X)) if i not in train_ind]

d = RVGP.create_data_object(X, faces, n_eigenpairs=n_eigenpairs)
d.random_vector_field(seed=1)
d.smooth_vector_field(t=100)

vector_field_kernel = ManifoldKernel(d, 
                                     nu=3/2, 
                                     kappa=3, 
                                     typ='matern',
                                     sigma_f=1.)

vector_field_GP = gpflow.models.GPR((d.evecs_Lc.reshape(d.n*vertices.shape[1], -1), 
                                     d.vectors.reshape(d.n*vertices.shape[1], -1)), 
                        kernel=vector_field_kernel, 
                        noise_variance=0.001,
                        )


K = vector_field_kernel(d.evecs_Lc)
n, dim = X.shape
K = K.numpy().reshape(n,dim,n,dim).swapaxes(1,2).swapaxes(2,3)
K = K[100]


ps.init()
ps_mesh = ps.register_surface_mesh("Surface points", vertices, faces)

ps_cloud = ps.register_point_cloud("Reference point", X[[100]])


ps_cloud_1 = ps.register_point_cloud("Points 1", X)
ps_cloud_1.add_vector_quantity("Kernel 1", K[:,:,0], color=(0, 0, 256), enabled=True)

ps_cloud_2 = ps.register_point_cloud("Points 2", X)
ps_cloud_2.add_vector_quantity("TKernel 2", K[:,:,1], color=(0, 256, 0), enabled=True)

ps_cloud_3 = ps.register_point_cloud("Points 3", X)
ps_cloud_3.add_vector_quantity("TKernel 3", K[:,:,2], color=(258, 0, 0), enabled=True)
ps.show()
