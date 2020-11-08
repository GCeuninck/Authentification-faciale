# -*- coding: utf-8 -*-
"""
Binome : DELECLUSE Raphael, CEUNINCK Guillaume 

source :
    https://www.codegrepper.com/code-examples/delphi/how+to+read+all+images+from+a+folder+in+python+using+opencv
"""

import os
import matplotlib.pyplot as plt
import random
import numpy as np


def load_images_from_folder(folder):
    """ 
    Chargement des images et noms d'images contenues dans un dossier.
    
    Entree:
        folder: Le dossier contenant les images a charger.
    
    Sortie:
        database: 
            La liste contenant chaque image initialement dans folder.
            Une image est representee par un np array.
        database_names: 
            La liste contenant les noms de chaque image initialement dans folder.
            Un nom est formate sous la forme X.Y.jpg 
	"""
    database = []
    database_names = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder,filename))
        if img is not None:
            database.append(img)
            database_names.append(filename.replace('.jpg', ''))
    return database, database_names

def generation_data(database, database_names, nb_probes = 100):
    """ 
    Generation de la gallery, des requetes connues et inconnues, 
    ainsi que les noms associes pour constituer la verite terrain. 
    
    Entree:
        database: 
            La liste contenant chaque image initialement dans folder.
            Une image est representee par un np array.
        database_names:
            La liste contenant les noms de chaque image initialement dans folder.
            Un nom est formate sous la forme X.Y.jpg
        nb_probes:
            Le nombre de requêtes connues et inconnues desire.
    
    Sortie:
        data:
            La liste ou la gallery d'images restantes apres suppression des requetes supposees inconnues.
            Une image est representee par un np array.
        data_names:
            La liste contenant les noms de chaque image de la gallery.
            Un nom est formate sous la forme X.Y
        probes_unknown:
            La liste des requetes supposees inconnues tirees aleatoirement.
            Une image est representee par un np array.
        names_unknown:
            La liste contenant les noms de chaque requetes supposees inconnues.
            Un nom est formate sous la forme X.Y
        probes_known:
            La liste des requetes supposees connues tirees aleatoirement.
            Une image est representee par un np array.
        names_known:
            La liste contenant les noms de chaque requetes supposees connues.
            Un nom est formate sous la forme X.Y
	"""
    
    data = database.copy()
    data_names = database_names.copy()
    probes_unknown, names_unknown, probes_known, names_known = [], [], [], []
    
    for i in range(nb_probes):
        integer = random.randint(0, len(data)-1)
        probes_unknown.append(data[integer])
        names_unknown.append(data_names[integer])
        
        indices = [i for i, elem in enumerate(data_names) if data_names[integer].split(".")[0] in elem]
        
        for y in range(len(indices)):
            data.pop(indices[0])
            data_names.pop(indices[0])
            
    for j in range(nb_probes):
        integer = random.randint(0, len(data)-1)
        probes_known.append(data[integer])
        names_known.append(data_names[integer])
        data.pop(integer)
        data_names.pop(integer)

    return data, data_names, probes_unknown, names_unknown, probes_known, names_known


def save_data(data, data_names, probes_unknown, names_unknown, probes_known, names_known):
    """ 
    Sauvegarde des informations de la gallery, des requetes connues et inconnues
    
    Entree:
        data:
            La liste ou la gallery d'images restantes apres suppression des requetes supposees inconnues.
            Une image est representee par un np array.
        data_names:
            La liste contenant les noms de chaque image de la gallery.
            Un nom est formate sous la forme X.Y
        probes_unknown:
            La liste des requetes supposees inconnues tirees aleatoirement.
            Une image est representee par un np array.
        names_unknown:
            La liste contenant les noms de chaque requetes supposees inconnues.
            Un nom est formate sous la forme X.Y
        probes_known:
            La liste des requetes supposees connues tirees aleatoirement.
            Une image est representee par un np array.
        names_known:
            La liste contenant les noms de chaque requetes supposees connues.
            Un nom est formate sous la forme X.Y
    
    Sortie:
        data.npy: 
            Le fichier contenant un np array de la gallery et des noms de chaque image associees.
        unknown.npy: 
            Le fichier contenant un np array des requetes inconnues et des noms de chaque image associees. 
        known.npy:
            Le fichier contenant un np array des requetes connues et des noms de chaque image associees. 
	"""
    
    with open('data_npy/data.npy', 'wb') as f:
        np.save(f, np.array([data, data_names]))
        
    with open('data_npy/unknown.npy', 'wb') as f:
        np.save(f, np.array([probes_unknown, names_unknown]))
        
    with open('data_npy/known.npy', 'wb') as f:
        np.save(f, np.array([probes_known, names_known]))
        
def load_data():
    """ 
    Chargement des informations de la gallery, des requetes connues et inconnues.
    
    Sortie:
        data:
            La liste ou la gallery d'images restantes apres suppression des requetes supposees inconnues.
            Une image est representee par un np array.
        data_names:
            La liste contenant les noms de chaque image de la gallery.
            Un nom est formate sous la forme X.Y
        probes_unknown:
            La liste des requetes supposees inconnues tirees aleatoirement.
            Une image est representee par un np array.
        names_unknown:
            La liste contenant les noms de chaque requetes supposees inconnues.
            Un nom est formate sous la forme X.Y
        probes_known:
            La liste des requetes supposees connues tirees aleatoirement.
            Une image est representee par un np array.
        names_known:
            La liste contenant les noms de chaque requetes supposees connues.
            Un nom est formate sous la forme X.Y
    """
    with open('data_npy/data.npy', 'rb') as f:
        temp = np.load(f,  allow_pickle=True)
        data = temp[0]
        data_names = temp[1]
    with open('data_npy/unknown.npy', 'rb') as f:
        temp = np.load(f,  allow_pickle=True)
        probes_unknown = temp[0]
        names_unknown = temp[1]
    with open('data_npy/known.npy', 'rb') as f:
        temp = np.load(f,  allow_pickle=True)
        probes_known = temp[0]
        names_known = temp[1]
        
    return data, data_names, probes_unknown, names_unknown, probes_known, names_known