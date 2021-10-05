import os
import numpy as np
import random
import argparse
import pickle

normalize_geometry_embeddings_list =[]

for n in range(960):

	lx = (8/960)*n + 0.5
	geometry_embeddings= [lx,3.5,1.5,8.8,3.5,1.5,9,7,3,0.35]
	max_dimension = 5
	normalize_geometry_embeddings =np.divide(geometry_embeddings,max_dimension)-1
	normalize_geometry_embeddings_list.append(normalize_geometry_embeddings)


embeddings_pickle ="example1.pickle"
with open(embeddings_pickle, 'wb') as f:
    pickle.dump(normalize_geometry_embeddings_list, f, protocol=2)