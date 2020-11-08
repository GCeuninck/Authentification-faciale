# -*- coding: utf-8 -*-
"""
Binome : DELECLUSE Raphael, CEUNINCK Guillaume
"""

import numpy as np
import NN_brute_force as NN_BF
import Eigenfaces as EG
import Generation_data 
import Metrics as Mt

def is_authorised(data,probe,radius):
    boolean = False
    
    indices_probe = NN_BF.radius_search(data, probe, radius)
    if len(indices_probe) != 0:
        boolean = True
    return boolean

def test(radius, dataset = 1, rule = "Coude", rayon_max = 10**8, pas = 10**6, calc_speedup = 0):
    
    if dataset == 1:
        folder="data/dataset1/images"
    elif dataset == 2:
        folder="data/dataset2/images"

    print("---Generation donnees---\n")
    database, database_names = Generation_data.load_images_from_folder(folder)
    data, data_names, probes_unknown, names_unknown, probes_known, names_known = Generation_data.generation_data(database, database_names)
    Generation_data.save_data(data, data_names, probes_unknown, names_unknown, probes_known, names_known)

    print("---Reduction de la dimension des donnees---\n")
    D_reduced, w_significatif = EG.ACP_efficace(data, rule)

    probes_known_centered = EG.transform_data(probes_known)
    probes_known_reduced = EG.projection(probes_known_centered, w_significatif)
    
    probes_unknown_centered = EG.transform_data(probes_unknown)
    probes_unknown_reduced = EG.projection(probes_unknown_centered, w_significatif)
    
    print("Test probe connu :", is_authorised(D_reduced, probes_known_reduced[0], radius))
    print("Test probe inconnu :", is_authorised(D_reduced, probes_unknown_reduced[0], radius))
    print("---Metrics---\n")
    metrics = np.array(Mt.evaluation(D_reduced, data_names, names_known, names_unknown, probes_known_reduced, probes_unknown_reduced, pas, rayon_max))
    Mt.save_metrics(metrics)
    Mt.trace_metrics(metrics)
    

    if (calc_speedup != 0):
    
        with open('data_npy/metrics_brut_force.npy', 'rb') as f:
            metrics_brut_force = np.load(f,  allow_pickle=True)
        #Mt.trace_metrics(metrics_brut_force)
        
        with open('data_npy/metrics_inertia.npy', 'rb') as f:
            metrics_inertia = np.load(f,  allow_pickle=True)
        #Mt.trace_metrics(metrics)
        
        with open('data_npy/metrics_kaiser.npy', 'rb') as f:
            metrics_kaiser = np.load(f,  allow_pickle=True)
        #Mt.trace_metrics(metrics)
            
        with open('data_npy/metrics_coude.npy', 'rb') as f:
            metrics_coude = np.load(f,  allow_pickle=True)
        #Mt.trace_metrics(metrics_coude)
        
        Mt.speedup(metrics_brut_force,metrics_kaiser,metrics_inertia,metrics_coude)
    

test(radius = 2*10**6, rayon_max = 2*10**7, pas = )

