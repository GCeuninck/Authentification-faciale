# -*- coding: utf-8 -*-
"""
Binome : DELECLUSE Raphael, CEUNINCK Guillaume
"""

import numpy as np
import matplotlib.pyplot as plt

def linearisation(data):
    """ 
    Linearisation des donnees.
    
    Entree:
        data: 
            La liste d'images correspondant a la gallery, aux requetes connues ou inconnues.
            Une image est representee par un np array de taille (p,p).
    
    Sortie:
        data_linear:
            Le np array contenant l'ensemble des images linearisees en dimension p*p.
    """
    
    data_linear = np.zeros((len(data),data[0].shape[0]*data[0].shape[1]))
    
    for i, image in enumerate(data):
        data_linear[i] = np.reshape(image, image.shape[0]*image.shape[1])
     
    return data_linear

def centraliser(data):
    """ 
    Centralisation des donnees, revenant a soustraire le visage d’un individu moyen a chaque individu.
    
    Entree:
        data: 
            La liste d'images correspondant a la gallery.
            Une image est representee par un np array.
    
    Sortie:
        Les donnees centrees.
    """
    
    mean = data.mean(axis = 0)       
    return data[:]-mean

def transform_data(data):
    """ 
    Transformation des donnees, revenant lineariser et centrer les donnnees.
    
    Entree:
        data: 
            La liste d'images correspondant a la gallery.
            Une image est representee par un np array.
    
    Sortie:
        Les donnees linearisees et centrees.
    """
    
    return centraliser(linearisation(data))

def projection(data, w):
    """ 
    Projection des donnees sur le sous-espace vectoriel engendre par les w vecteurs.
    
    Entree:
        data: 
            La liste d'images correspondant a la gallery.
            Une image est representee par un np array.
        w:
            Les vecteurs engendrant le sous-espace vectoriel. 
    
    Sortie:
        Les donnees projetees sur le sous-espace vectoriel engendre par les w vecteurs.
    """
    
    return  data.dot(w)

def Calc_valeurs_vecteurs_propres(D):
    """ 
    Calcule des valeurs propres mu et des vecteurs w normalises associes a D
    
    Entree:
        D: 
            La liste d'images linearisees et centrees correspondant a la gallery.
            Une image est representee par un np array.
    
    Sortie:
        mu: 
            Les valeurs propres associees a D.
        w_norm:
            Les n vecteurs propres normalises associes a D.
    """
    
    D_t = D.transpose()
    cov = D.dot(D_t)/(D.shape[1]-1)
    values, vectors = np.linalg.eig(cov)
    w = D_t.dot(vectors)
        
    #valeurs propres de D :
    mu = ((D.shape[1] -1)/(D.shape[0] -1)) * values
    
    #Normalisation des vecteurs
    normes = np.linalg.norm(w, axis = 0)
    w_norm = w/normes
    
    return mu, w_norm


def ACP_efficace(data, rule = "Kaiser"):
    """ 
    Realisation de l'ACP efficace selon le critere rule
        
    Entree:
        data: 
            La liste d'images correspondant a la gallery.
            Une image est representee par un np array.
        rule:
            Le critere choisi pour selectionner les premiers vecteurs principaux.
            * Kaiser : Le critere de Kaiser est choisi pour selectionner les k premiers vecteurs principaux.
            * Inertia : Le critere d'inertie est choisi pour selectionner les k premiers vecteurs principaux.
            * Coude : Le critere d'eboulis des parts d’inertie est choisi pour selectionner les k premiers vecteurs principaux.
    
    Sortie:
        D_reduced: 
            Les donnees dont la dimension a ete reduites sur le sous-espaces engendre par les k premiers vecteurs principaux.
        w_significatif:
            Les k premiers vecteurs principaux permettant la reduction de la dimension des donnees.
    """
    
    #Linéarisation et centralisation par rapport aux variables
    D = transform_data(data)
    
    #Calcul des valeurs propres et vecteurs propres de D
    mu, w_norm = Calc_valeurs_vecteurs_propres(D)
    
    #Selections des K vecteurs principaux
    w_significatif = select_significatif_vectors(mu, w_norm, rule)
    
    #Projection des données
    D_reduced = projection(D,w_significatif)
    
    return D_reduced, w_significatif

  
def select_significatif_vectors(mu, w, rule = "Kaiser"):
    """ 
    Retourne les k premiers vecteurs principaux selon le critere rule choisi.
        
    Entree:
        mu: 
            Les valeurs propres associees a D.
        w:
            Les vecteurs propres normalises associes a D.
        rule:
            Le critere choisi pour selectionner les premiers vecteurs principaux.
            * Kaiser : Le critere de Kaiser est choisi pour selectionner les k premiers vecteurs principaux.
            * Inertia : Le critere d'inertie est choisi pour selectionner les k premiers vecteurs principaux.
            * Coude : Le critere d'eboulis des parts d’inertie est choisi pour selectionner les k premiers vecteurs principaux.
        
    Sortie:
        Les k premiers vecteurs principaux permettant la reduction de la dimension des donnees.
    """
    
    #Régle de Kaiser 
    #Les valeurs propres sont triés par ordre décroissant
    
    if rule == "Kaiser":
        index_significatif = []
        i = 0
        Inertie_moy = (np.sum(mu)/w.shape[0])
        
        while mu[i]>=Inertie_moy:
            index_significatif.append(i)
            i+=1
            
    #Régle de la part d'inertie
    #On cherche à conserver 80% de l'inertie initiale
        
    if rule == "Inertia":
        index_significatif = []
        i = 0
        Inertie_tot = np.sum(mu)
        
        while np.sum(mu[:i])/Inertie_tot <= 0.8:
            index_significatif.append(i)
            i+=1
    
    if rule == "Coude":
        index_significatif = []
        i = 0
        
        while i < 10:
            index_significatif.append(i)
            i+=1
        
    return w[:,index_significatif]


def retrieved_image(D, eigenvectors, moy):
    """ 
    Retrouve les images projetees.
    Une image sera sous la forme (p,p).
        
    Entree:
        D: 
            Les donnees dont la dimension a ete reduites sur le sous-espaces engendre par les k premiers vecteurs principaux.
        eigenvectors:
            Les k premiers vecteurs principaux de D.
        moy:
            Le visage d’un individu moyen de D.
    Sortie:
        data_linear:
            Le np array des images projetees sous la forme (p,p).
    """
    
    Y = eigenvectors.dot(D.transpose())
    
    data = Y.transpose() + moy
    
    image_size = int(np.sqrt(len(data[0])))
    
    data_linear = np.zeros((len(data),image_size,image_size))
    
    for i, image in enumerate(data):
        data_linear[i] = np.reshape(image, (image_size,image_size))

    return data_linear

def critere_coude(data, echelle):
    """ 
    Generation du graphique representant la part de l'inertie totale en % en fonction des k premiers vecteurs principaux.
    
    Entree:
        data: 
            La liste d'images correspondant a la gallery.
            Une image est representee par un np array.
        echelle:
            Selectionne les k premiers vecteurs principaux.
         
    Sortie:
        Graphique representant la part de l'inertie totale en % en fonction du nombre de premiers vecteurs principaux k.
    """
    
    D = transform_data(data)
    mu = Calc_valeurs_vecteurs_propres(D)[0]

    mu_Inertie = (mu/np.sum(mu))*100
    
    fig_coud, ax_coud = plt.subplots()
    ax_coud.plot(mu_Inertie[:echelle])
    
    plt.xlabel("Nombre de vecteurs propres")
    plt.ylabel("Part de l'inertie totale en %")
    plt.title(" Eboulis des parts d'inertie")
