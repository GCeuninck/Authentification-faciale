# -*- coding: utf-8 -*-
"""
Binome : DELECLUSE Raphael, CEUNINCK Guillaume
"""

import numpy as np

def euclidean_distances(data, q):
    """ 
    Calcule les distances euclidiennes entre chaque image de la gallery et une requete.
    
    Entree:
        data: 
            La liste ou la gallery d'images.
            Une image est representee par un np array. 
        q: 
            La requete dont on cherche a calculer les distances qui la separent de chaque image.
            Une image est representee par un np array.
    
    Sortie:
        distances:
            Le np array contenant l'ensemble des distances qui separent la requete de chaque image de la gallery.
	"""
    distances = np.zeros(len(data))
    for i, d in enumerate(data):
        distances[i] = np.sum((d-q)**2)
    return distances
    
def radius_search(data, q, r):
	""" 
    Retourne la liste des indices des plus proches voisins de la requete q dont la distance est inferieure au rayon r.
    
    Entree:
        data: 
            La liste ou la gallery d'images.
            Une image est representee par un np array. 
        q: 
            La requete dont on cherche a calculer les distances qui la separent de chaque image.
            Une image est representee par un np array.
        r:
            Le rayon delimitant la distance maximale requise pour etre considere comme l'un des plus proches voisins.
    
    Sortie:
        indices:
            La liste des indices plus proches voisins de la requete q selon r.
    """
    
	distances = euclidean_distances(data, q)
	indices = np.where(distances <= r)[0]
	return indices

def NN_bf_search(data, queries, r):
    """ 
    Retourne la liste des indices des plus proches voisins sur l'ensemble des requetes, dont la distance est inferieure au rayon r.
    
    Entree:
        data: 
            La liste ou la gallery d'images.
            Une image est representee par un np array. 
        queries: 
            La liste des requetes dont on cherche a calculer les distances qui les separent de chaque image.
            Une image est representee par un np array.
        r:
            Le rayon delimitant la distance maximale requise pour etre considere comme l'un des plus proches voisins.
    
    Sortie:
        indices:
            La liste des indices des plus proches voisins sur l'ensemble des requetes, selon r.
    """
    indices = []

    for i, q in enumerate(queries):
        ind = radius_search(data, q, r)
        indices.append(ind)
    
    return indices
