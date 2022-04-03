
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:34:53 2022

@author: MohammedZ
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

#%% importer le fichier
def load_data(filename): # methode donnée
    X, y = [], []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            X.append(row[:-1])
            y.append(row[-1])
    return np.array(X).astype(float), np.array(y)

iris = load_data("irisV1.csv")
x=pd.DataFrame(iris[0])
y=pd.DataFrame(iris[1])

#%% une classe KMeans pour generer un clustering
class KMeans:
    # constructeur avec pramaetres : par defaut K-2 et distance Euclidienne (0) comme sur Weka
    # 0 - euclidienne
    # 1 - manhattan
    # xCol de 2 car pour IrisV1 on veut utiliser la colonne 2 (petal_length)
    # yCol de 3 car pour IrisV1 on veut utiliser la colonne 3 (petal_width)
    def __init__(self, X, K=2, distanceType=0, xCol= 2, yCol=3): 
        self.K = K
        self.distanceType = distanceType
        self.sampleCount = X.shape[0]
        self.variableCount = X.shape[1]
        self.maxIterations = 250
        self.xCol = xCol
        self.yCol = yCol
        
    # methode principale de KMeans (comme donné dans le fichier d'exemple sur dvo)
    def fit(self, X):
        centroides = self.init_random_centroides(X) # créer K centroides aléatoirement
        dispersions = {} # On recupere les differentes clusters de chaque iteration de Dispersion pour enfin choisir la meilleure IntraClasse
        for rep in range(10): # Faire  fois la repetition, en calculant les dispersions de chaque clusters
            for it in range(self.maxIterations): # un limiteur de boucle au cas ou
                clusters = self.create_clusters(X, centroides)
                centroides_prec = centroides # on met le centroide en memoire, pour comparer avec la prochaine iteration si il y a une variation
                centroides = self.create_new_centroides(X, clusters) # créér des centroides selon les nouveaux clusters
                diff = centroides - centroides_prec # comparaisons entre centroides actuel et précedents
                if not diff.any(): # si il N'ya pas de differences, alors on arrete cette itération
                    disp = self.dispersionIntra(X, clusters, centroides) # on calucle la dispersion entre les differentes clusters
                    dispersions[disp] = clusters 
                    break
        clusters = dispersions[min(dispersions)] # on prends le cluster qui a la plus petite valeur de dispersion intraclasse
        y_predict = self.clusters_Finale(X, clusters) # un tableau [0,3,1,0,1,2,2] qui signifie les differentes classes (donc couleur)
        self.plot_fig(X, y_predict) # afficher tableau (console)
        return y_predict
    
    # methode pour créer K centroides aléatoirement
    def init_random_centroides(self, X):
        centroides = [[0 for x in range(self.variableCount)] for y in range(self.K)]  # créer une matrice nul (similaire au tableau de centroides de TD)
        for k in range(self.K): 
            centroide = X.iloc[np.random.choice(range(self.sampleCount))] # choisir un point aléatoirement dans les données, ce point sera notre centroidé temporaire
            centroides[k] = centroide # ajouter ce centroide dans la liste des centroides
        return centroides
    
    # methode pour créer K centroides aléatoirement (selon les etapes vu en TD)
    def init_random_centroides2(self, X):
        centroides = [[0 for x in range(self.variableCount)] for y in range(self.K)]  # créer une matrice nul (similaire au tableau de centroides de TD)
        self.classes = [[] for i in range(self.K)] # ex: k=3 : [[classe1], [classe2], [classe3]]
        T = X.to_numpy() # pas obligé mais c'est plus facile de travailler avec une liste numpy qu'un dataframe
        # on va attribuer les differents points dans une classe aléatoirment comme fait en TD
        for point_idx, point in enumerate(T): # iterer chaque ligne (point) 
            self.classes[np.random.choice(range(self.K))].append(point)
        # on crée les centroides en trouvant la moyenne des colonnes dans les classes
        for k in range(self.K): 
            centroid = np.average(self.classes[k], axis=0)
            centroides[k] = centroid
        return centroides
    
    def distanceEuclidienne(self, p1, p2):
        # ex : (0-2)²  + (2-2)² + (4-2)²  = 8
        return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
    
    def distanceManhattan(self, p1, p2):
        dist = np.sum(np.abs(value1 - value2) for value1, value2 in zip(p1, p2))
        return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
    
    # methode pour generer les clusters (une liste qui contient les points)
    def create_clusters(self, X, centroides):
        clusters = [[] for i in range(self.K)] 
        T = X.to_numpy() # plus simple à travailler avec numpy
        for point_idx, point in enumerate(T):
            # ici on veut trouver pour chaque point, le centroide le plus proche donc np.argmin()
            if(self.distanceType == 0): # euclidienne
                closest_centroid = np.argmin(self.distanceEuclidienne(point, centroides)) # trouver le centroide le plus proche à ce point
            else: # cas distance Manhattan
                closest_centroid = np.argmin(self.distanceManhattan(point, centroides))
            
            clusters[closest_centroid].append(point_idx) #ajouter ce point dans le cluster correspondant
        return clusters
    
    # methode pour créer une liste contenant les K nouveaux centroides selon les points dans les clusters
    def create_new_centroides(self, X, clusters):
        centroides = np.zeros((self.K, self.variableCount)) # créer une matrice nul
        for idx, cluster in enumerate(clusters):
            new_centroide = np.mean(X.iloc[cluster], axis=0) # prendre la moyenne de cluster pour trouver la position de nouveau centroide
            centroides[idx] = new_centroide
        return centroides
    
    # methode qui créer une liste de type [0, 2, 1, 4....] qui sont les valeurs de classe finale pour chaque points de donées
    def clusters_Finale(self, X, clusters): # generate ypredict which has 0,1,2.. as result for colors
        y_predict = [0 for y in range(self.sampleCount)] # [0,0,0,...0]
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_predict[sample_idx] = cluster_idx
        return y_predict
    
    # methode pour afficher le graphe (sur console)
    def plot_fig(self, X, y):
        xAxis = self.xCol - self.variableCount
        yAxis = self.yCol - self.variableCount
        plt.scatter(X.iloc[:,xAxis], X.iloc[:,yAxis], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
        
    # methode helper crée pour utiliser plus tard (tkinter)
    def getVariableCount(self):
        return int(self.variableCount)
    
    # Methode pour calculer la dispersion Intra-Classe dans un cluster
    def dispersionIntra(self, X, clusters, centroides):
        dispersions = []
        T = X.to_numpy()
        for cId, cluster in enumerate(clusters):
            clusterX = np.array([T[i] for i in cluster])
            clusterSum = 0
            for col in range(self.variableCount):
                clusterSum += np.sum((clusterX[:,col] - centroides[cId][col])**2, axis=0)
            dispersions.append(clusterSum)
        variance = np.var(dispersions) # variance entre les differents classes d'un cluster
        #print(variance)
        return variance
        
#%% Initialization Kmeans
#model=KMeans(x, 2)
#y_pred = model.fit(x)  
        
#%% Tkinter
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

fenetre = Tk()
fenetre.title('KMeans Simulator')
fenetre.geometry("840x550")

fichierLbl = Label(fenetre, text="Nom du fichier", font=('Arial', 12))
fichierLbl.grid(row=0, column=0)
nomFichier = Entry(fenetre, width=22, font=('Arial 13'), borderwidth=1)
nomFichier.insert(END, 'IrisV1.csv')
nomFichier.grid(row=1, column=0, columnspan=3)

KclustLbl = Label(fenetre, text="K (nbr. clusters)", font=('Arial', 12))
KclustLbl.grid(row=2, column=0)
Kclust = Entry(fenetre, width=22, font=('Arial 13'), borderwidth=1)
Kclust.insert(END, '2')
Kclust.grid(row=3, column=0, columnspan=3)

distanceLbl = Label(fenetre, text="Distance", font=('Arial', 12))
distanceLbl.grid(row=4, column=0)
distance = ttk.Combobox(fenetre,width=30, values=["Euclidienne", "Manhattan"])
distance.current(0)
distance.grid(row=5, column=0)

col1Lbl = Label(fenetre, text="X Axis Column", font=('Arial', 12))
col1Lbl.grid(row=6, column=0)
col1 = Entry(fenetre, width=22, font=('Arial 13'), borderwidth=1)
col1.insert(END, '2')
col1.grid(row=7, column=0, columnspan=3)

col2Lbl = Label(fenetre, text="Y Axis Column", font=('Arial', 12))
col2Lbl.grid(row=8, column=0)
col2 = Entry(fenetre, width=22, font=('Arial 13'), borderwidth=1)
col2.insert(END, '3')
col2.grid(row=9, column=0, columnspan=3)

def plotKMeans():
    fichier = nomFichier.get()
    k = int(Kclust.get())
    xCol, yCol = int(col1.get()), int(col2.get())
    iris = load_data(fichier)
    x=pd.DataFrame(iris[0])
    y=pd.DataFrame(iris[1])
    figure = Figure(figsize=(5, 5), dpi=105)
    ax1 = figure.add_subplot(111)
    model=KMeans(x, k, xCol = xCol, yCol = yCol)
    y_pred = model.fit(x)
    xAxis = xCol - model.getVariableCount()
    yAxis = yCol - model.getVariableCount()
    ax1.scatter(x.iloc[:,xAxis], x.iloc[:,yAxis], c=y_pred, s=40, cmap=plt.cm.Spectral)
    figure_canvas = FigureCanvasTkAgg(figure, fenetre)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().grid(row=0, rowspan=11, column=3, padx=10, pady=10)
    

generate = Button(fenetre, text="Plot KMeans",bd=0, padx=100,bg="lightgreen", pady=10, command=lambda: plotKMeans())
generate.grid(row=10, column=0, pady=7)


figure = Figure(figsize=(5, 5), dpi=105)
ax1 = figure.add_subplot(111)
model=KMeans(x, 2, xCol = 2, yCol = 3)
y_pred = model.fit(x)
xAxis = 2 - model.getVariableCount()
yAxis = 3 - model.getVariableCount()
ax1.scatter(x.iloc[:,xAxis], x.iloc[:,yAxis], c=y_pred, s=40, cmap=plt.cm.Spectral)
figure_canvas = FigureCanvasTkAgg(figure, fenetre)
figure_canvas.draw()
figure_canvas.get_tk_widget().grid(row=0, rowspan=11, column=3, padx=10, pady=10)


fenetre.mainloop()
        
        
        
        
        
        
        
        