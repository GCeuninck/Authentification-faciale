# -*- coding: utf-8 -*-
"""
Binome : DELECLUSE Raphael, CEUNINCK Guillaume
"""

import NN_brute_force as NN_BF
import numpy as np
import matplotlib.pyplot as plt
import time

def mesure_cas(data_names, names_known, names_unknown, indices_probes_known, indices_probes_unknown):
    """ 
    Mesure des nombres de True Positif, True Negatif, False Positiv, False Negativ, necessaires pour le calcul des metriques.
    
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
        TP:
            Nombre de fois où le système autorise l'accès à un utilisateur auquel l'accès doit être autorisé pour des motifs légitimes
        TN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être refusé.
        FP:
            Nombre de fois où le système autorise l'accès à tort. 
        FN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être autorisé.
    """
    
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for i, probes in enumerate(indices_probes_known):
        if probes.shape[-1] == 0:
            FN +=1
        else:
            stop = 0
            index_NN = 0
            while stop == 0 and index_NN < len(probes):
                if data_names[probes[index_NN]].split(".")[0] == names_known[i].split(".")[0]:
                    TP +=1
                    stop = 1
                index_NN += 1
            if stop == 0:
                FP +=1
                
    #Pour les unknown :
    
    for y in indices_probes_unknown:
        #Si le tableau est vide, alors on n'a rien trouvé : Vrai négatif
        if y.shape[-1] == 0:
            TN +=1
        #Sinon, on a trouvé une image pour un utilisateur non-enregistré : Faux positif
        else :
            FP +=1
    
    return TP, TN, FP, FN

def calc_metrics(D, data_names, names_known, names_unknown, probes_known, probes_unknown, radius):
    """ 
    Retourne la liste de l'evaluation des performances a l'aide de differentes metriques pour un radius de plus proches voisins choisi.
    
    Entree:
        D:
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
        radius:
            Le rayon delimitant la distance maximale requise pour etre considere comme l'un des plus proches voisins.
    
    Sortie:
        La liste contenant le radius et les performances suivantes:
        *exac: Mesure le taux de bonnes reponses produites par le systeme.
        *prec: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
        *rapp: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
        *spec: Mesure le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
        *tps_de_recherche: Le temps qu'a pris le systeme pour trouver les plus proches voisins des requetes connues et inconnues.
    """
    
    startTime = time.time()
    
    indices_probes_known = NN_BF.NN_bf_search(D, probes_known, radius)
    indices_probes_unknown = NN_BF.NN_bf_search(D, probes_unknown, radius)
    
    tps_de_recherche = time.time() - startTime
    
    TP, TN, FP, FN = mesure_cas(data_names, names_known, names_unknown, indices_probes_known, indices_probes_unknown)
    
    exac = exactitude(TP, TN, FP, FN)
    prec = precision(TP, FP)
    rapp = rappel(TP, FN)
    spec = specificite(TN, FP)
        
    return [radius,exac,prec,rapp,spec,tps_de_recherche]

def evaluation(D_reduced, data_names, names_known, names_unknown, probes_known_reduced, probes_unknown_reduced, pas, radius_max):
    """ 
    Retourne la liste de l'ensemble des evaluations des performances a l'aide de differentes metriques 
    pour un radius de plus proches voisins variant de 0 a radius_max avec un certain pas.
    
    Entree:
        D_reduced:
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
        radius_max:
            La borne maximale que le radius va pouvoir varier.
        pas:
            Le pas pour lequel le radius va etre incremente.
    
    Sortie:
        La liste contenant l'ensemble des radius testes et performances suivantes:
        *exac: Mesure le taux de bonnes reponses produites par le systeme.
        *prec: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
        *rapp: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
        *spec: Mesure le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
        *tps_de_recherche: Le temps qu'a pris le systeme pour trouver les plus proches voisins des requetes connues et inconnues.
    """
    
    res = []
    
    for r in range(0, radius_max, pas):
        
        metrics = calc_metrics(D_reduced, data_names, names_known, names_unknown, probes_known_reduced, probes_unknown_reduced, r)
        
        res.append(metrics)
        
    return res

def exactitude(TP,TN,FP,FN):
    """ 
    Mesure la metrique d'exactitude, soit le taux de bonnes reponses produites par le systeme.
    
    Entree:
        TP:
            Nombre de fois où le système autorise l'accès à un utilisateur auquel l'accès doit être autorisé pour des motifs légitimes.
        TN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être refusé.
        FP:
            Nombre de fois où le système autorise l'accès à tort. 
        FN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être autorisé.
    
    Sortie:
        exactitude:
            Le resultat de la metrique calculee.
    """
    
    exactitude = 0
    if (TP+FP+TN+FN) != 0:
        exactitude = (TP+TN)/(TP+FP+TN+FN)
    return exactitude

def precision(TP, FP):
    """ 
    Mesure la metrique de precision, soit le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
    
    Entree:
        TP:
            Nombre de fois où le système autorise l'accès à un utilisateur auquel l'accès doit être autorisé pour des motifs légitimes.
        TN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être refusé.
        FP:
            Nombre de fois où le système autorise l'accès à tort. 
        FN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être autorisé.
    
    Sortie:
        precision:
            Le resultat de la metrique calculee.
    """
    precision = 0
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    return precision

def rappel(TP,FN):
    """ 
    Mesure la metrique de rappel, soit le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
    
    Entree:
        TP:
            Nombre de fois où le système autorise l'accès à un utilisateur auquel l'accès doit être autorisé pour des motifs légitimes.
        TN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être refusé.
        FP:
            Nombre de fois où le système autorise l'accès à tort. 
        FN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être autorisé.
    
    Sortie:
        rappel:
            Le resultat de la metrique calculee.
    """
    rappel = 0
    if (TP+FN) != 0:
        rappel = TP/(TP+FN)
    return rappel

def specificite(TN, FP):
    """ 
    Mesure la metrique de specificite, soit le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
    
    Entree:
        TP:
            Nombre de fois où le système autorise l'accès à un utilisateur auquel l'accès doit être autorisé pour des motifs légitimes
        TN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être refusé.
        FP:
            Nombre de fois où le système autorise l'accès à tort. 
        FN:
            Nombre de fois où le système refuse l'accès à un utilisateur auquel l'accès doit être autorisé.
    
    Sortie:
        specificite:
            Le resultat de la metrique calculee.
    """
    specificite = 0
    if(TN+FP) != 0:
        specificite = TN /(TN+FP)
    return specificite

def save_metrics(metrics):
    """ 
    Sauvegarde des informations des metrics pour les differents radius.
    
    Entree:
        metrics:
            La liste contenant l'ensemble des radius testes et performances suivantes:
            *exac: Mesure le taux de bonnes reponses produites par le systeme.
            *prec: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
            *rapp: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
            *spec: Mesure le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
            *tps_de_recherche: Le temps qu'a pris le systeme pour trouver les plus proches voisins des requetes connues et inconnues.
    
    Sortie:
        metrics.npy: 
            Le fichier contenant un np array des informations des metrics pour les differents radius.
    """
    
    with open('data_npy/metrics.npy', 'wb') as f:
        np.save(f, np.array(metrics))
        
def load_metrics():
    """ 
    Chargement des informations des metrics pour les differents radius.
    
    Sortie:
        metrics:
            La liste contenant l'ensemble des radius testes et performances suivantes:
            *exac: Mesure le taux de bonnes reponses produites par le systeme.
            *prec: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
            *rapp: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
            *spec: Mesure le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
            *tps_de_recherche: Le temps qu'a pris le systeme pour trouver les plus proches voisins des requetes connues et inconnues.
    """
    with open('data_npy/metrics.npy', 'rb') as f:
        metrics = np.load(f,  allow_pickle=True)
    return metrics

def trace_metrics(metrics):
    """ 
    Generation des graphiques representant l'evolution des differentes metriques et du temps d'execution en fonction du radius, ainsi que des courbes de ROC.
    
    Entree:
        metrics:
            La liste contenant l'ensemble des radius testes et performances suivantes:
            *exac: Mesure le taux de bonnes reponses produites par le systeme.
            *prec: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes autorisees par le systeme.
            *rapp: Mesure le taux de personnes correctement autorisees parmi l'ensemble des personnes qui devraient etre autorisees par le systeme.
            *spec: Mesure le taux de personnes correctement refusees parmi l'ensemble des personnes refusees par le systeme.
            *tps_de_recherche: Le temps qu'a pris le systeme pour trouver les plus proches voisins des requetes connues et inconnues.
    Sortie:
        Graphiques representant l'evolution des differentes metriques et du temps d'execution en fonction du radius, ainsi que des courbes de ROC.
    """
    
    fig_exac, ax_exac = plt.subplots()
    ax_exac.plot(metrics[:,0],metrics[:,1])
    ax_exac.xaxis.set_ticks(np.arange(metrics[0,0], metrics[-1,0], 10**7))
    plt.xlabel("R")
    plt.ylabel("Exactitude")
    plt.title("Exactitude par rapport a R")
    
    fig_prec, ax_prec = plt.subplots()
    ax_prec.plot(metrics[:,0],metrics[:,2])
    ax_prec.xaxis.set_ticks(np.arange(metrics[0,0], metrics[-1,0], 10**7))
    plt.xlabel("R")
    plt.ylabel("Precision")
    plt.title("Precision par rapport a R")
    
    fig_rapp, ax_rapp = plt.subplots()
    ax_rapp.plot(metrics[:,0],metrics[:,3])
    ax_rapp.xaxis.set_ticks(np.arange(metrics[0,0], metrics[-1,0], 10**7))
    plt.xlabel("R")
    plt.ylabel("Rappel")
    plt.title("Rappel par rapport a R")
    
    fig_spec, ax_spec = plt.subplots()
    ax_spec.plot(metrics[:,0],metrics[:,4])
    ax_spec.xaxis.set_ticks(np.arange(metrics[0,0], metrics[-1,0], 10**7))
    plt.xlabel("R")
    plt.ylabel("Specificite")
    plt.title("Specificite par rapport a R")
    
    fig_roc, ax_roc = plt.subplots(figsize = (5,5))
    x,y = np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1)
    
    ax_roc.plot(1 - metrics[:,4],metrics[:,3], label = "ROC curve")
    ax_roc.plot(0, 1, "ro", label="Perfect classification")
    ax_roc.plot(x, y, ":", label="Random guess curve")
    ax_roc.xaxis.set_ticks(x)
    ax_roc.yaxis.set_ticks(y)
    ax_roc.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe de ROC")
    
    fig_tps, ax_tps = plt.subplots()
    ax_tps.plot(metrics[:,0],metrics[:,5])
    ax_tps.xaxis.set_ticks(np.arange(metrics[0,0], metrics[-1,0], 10**7))
    #ax_tps.yaxis.set_ticks(np.arange(5, 8, 1))
    plt.xlabel("R")
    plt.ylabel("Temps d'execution")
    plt.title("Temps d'execution par rapport a R")
     
def speedup(metrics_bf,metrics_kaiser,metrics_inertie,metrics_coude):
    """ 
    Generation du graphique representant l'evolution du speedup en fonction du radius, pour les trois systemes suivants:
    *Système d'authentification avec Eigenfaces et critère de Kaiser.
    *Système d'authentification avec Eigenfaces et critère d’inertie.
    *Système d'authentification avec Eigenfaces et critère d’éboulis.
    
    Entree:
        metrics_bf:
            La liste contenant l'ensemble des radius testes et performances mesurees pour la methode sans ACP
        metrics_kaiser:
            La liste contenant l'ensemble des radius testes et performances mesurees pour la methode avec critere de Kaiser
        metrics_inertie:
            La liste contenant l'ensemble des radius testes et performances mesurees pour la methode avec critere d'inertie
        metrics_coude:
            La liste contenant l'ensemble des radius testes et performances mesurees pour la methode critere d'eboulis
    Sortie:
        Graphique representant l'evolution du speedup en fonction du radius, pour les trois systemes.
    """
    
    fig_tps, ax_tps = plt.subplots()
    ax_tps.plot(metrics_bf[:,0],metrics_bf[:,5]/metrics_inertie[:,5],label="Taux d’inertie cumule")
    ax_tps.plot(metrics_bf[:,0],metrics_bf[:,5]/metrics_kaiser[:,5],label="Critere de Kaiser")
    ax_tps.plot(metrics_bf[:,0],metrics_bf[:,5]/metrics_coude[:,5],label="Eboulis des parts d’inertie")
    ax_tps.legend()
    ax_tps.xaxis.set_ticks(np.arange(metrics_bf[0,0], metrics_bf[-1,0], 10**7))
    #ax_tps.yaxis.set_ticks(np.arange(5, 8, 1))
    plt.xlabel("R")
    plt.ylabel("Speedup")
    plt.title("Speedup par rapport a R")
    