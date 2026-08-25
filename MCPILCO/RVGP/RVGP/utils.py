#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

def load_mesh(data='bunny', folder=None):
    vertices = []
    faces = []
    
    dirname = os.path.dirname(os.path.realpath(__file__))
    if folder is None:
        file = os.path.join(dirname, '..', 'examples/data', data)
    else:
        file = os.path.join(folder, data)
    
    with open('{}.obj'.format(file), 'r') as file:

        for line in file:

            if line.startswith('#'):
                continue

            words = line.split()

            if words[0] == 'v':
                vertex = [float(words[1]), float(words[2]), float(words[3])]
                vertices.append(vertex)
            elif words[0] == 'f':
                face = [int(words[1]), int(words[2]), int(words[3])]
                faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)-1
    
    return vertices, faces
