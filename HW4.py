# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:52:02 2015

@author: DAN
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random


prob1Data = None
prob1Data2 = None
clusterCenters = None



def prob1():
    global prob1Data
    global clusterCenters
    global prob1Data2    
    
    listdat = [(6, 12), (19, 7), (15, 4), (11, 0), 
                              (18, 12), (9, 20), (19, 22), (18, 17), 
                            (5, 11), (4, 18), (7, 15), (21, 18), (1, 19), 
                            (1, 4), (0, 9), (5, 11)]    
    
    #Initialize the data
    prob1Data = pd.DataFrame(listdat, columns=['X','Y'])
    prob1Data['Cluster'] = pd.Series(np.zeros(prob1Data.shape[0]))
    #Reverse the data order and create a new data set for it.
    listdat.reverse()
    prob1Data2 = pd.DataFrame(listdat, columns=['X','Y'])
    prob1Data2['Cluster'] = pd.Series(np.zeros(prob1Data.shape[0]))
    
    #randomIndex = np.arange(prob1Data.shape[0])
    #np.random.shuffle(randomIndex)
    #prob1Data2 = prob1Data2.loc[randomIndex]

    #RUN PART A
    clusterCenters = sequantialClusteringAlgorithm(prob1Data)
    #Plot the data on a scatter plot. 
    colorHandles = []
    #Make a list to hold the plotted cluster radi.
    clusterCircles = []
    #Create each cluster scatter plot.
    for cluster in range(1, 5):
        randColor = [random.random(), random.random(), random.random()]
        colorHandles.append(matplotlib.patches.Patch(color=randColor, label='K=' + str(cluster)))      
        plt.scatter(prob1Data[prob1Data.Cluster == cluster].X, prob1Data[prob1Data.Cluster == cluster].Y, label="Cluster " + str(cluster), color=randColor)     
        clusterCircles.append(plt.Circle((clusterCenters[cluster][0], clusterCenters[cluster][1]), 12, color=randColor, fill=False, clip_on=False))
    #plt.scatter(prob1Data[prob1Data.Cluster == 1].X, prob1Data[prob1Data.Cluster == 1].Y, label='Cluster 1', color=)     
    plt.legend(handles=colorHandles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    #Plot cluster circles with radius 12
    #fig = plt.gcf()
    #for circle in clusterCircles:
    #    fig.gca().add_artist(circle)   
    plt.show()
    plt.clf() 
    
    #RUN PART B
    clusterCenters2 = sequantialClusteringAlgorithm(prob1Data2)
    #Plot the data on a scatter plot. 
    colorHandles = []
    #Create each cluster scatter plot.
    for cluster in range(1, 5):
        randColor = [random.random(), random.random(), random.random()]
        colorHandles.append(matplotlib.patches.Patch(color=randColor, label='K=' + str(cluster)))      
        plt.scatter(prob1Data2[prob1Data2.Cluster == cluster].X, prob1Data2[prob1Data2.Cluster == cluster].Y, label="Cluster " + str(cluster), color=randColor)     
    plt.legend(handles=colorHandles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  
    plt.show()
    


'''
Clusters 1, 2, and 3.

Using incremental average newave = oldave + (an−oldave)/n.
'''
def sequantialClusteringAlgorithm(dataSet):
    clusterCenters = {}
    #Algorithm Parameters. 
    theta = 12
    maxClusters = 4
    numClusters = 1

    #Set first point to its own cluster. 
    #For each data point
    #1) Calculate distance to closest cluster center
    #2) If dist < alg and numClusters < 4
        #Create new cluster with data point.
    #3) Else, add data point to cluster. 

    clusterCenters[numClusters] = (dataSet.ix[0].X, dataSet.ix[0].Y, 1)
    dataSet['Cluster'].loc[0] = numClusters

    for index in range(1, dataSet.shape[0]):
        distance = 999999999
        clusterToAssign = 0
        for cluster in clusterCenters.keys():
            tempDist = dist(dataSet.ix[index], clusterCenters[cluster])
            if tempDist < distance:
                distance = tempDist
                clusterToAssign = cluster
        if distance <= theta and numClusters <= maxClusters:
            dataSet['Cluster'].loc[index] = clusterToAssign 
            newX = clusterCenters[clusterToAssign][0] + (dataSet.ix[index].X - clusterCenters[clusterToAssign][0])/(clusterCenters[clusterToAssign][2]+1.0) 
            newY = clusterCenters[clusterToAssign][1] + (dataSet.ix[index].Y - clusterCenters[clusterToAssign][1])/(clusterCenters[clusterToAssign][2]+1.0) 
            newSize = clusterCenters[clusterToAssign][2] + 1
            clusterCenters[clusterToAssign] = (newX, newY, newSize)
            dataSet['Cluster'].loc[index] = clusterToAssign
        elif distance > theta and numClusters < maxClusters:
            numClusters += 1
            dataSet['Cluster'].loc[index] = numClusters
            clusterCenters[numClusters] = (dataSet.ix[index].X, dataSet.ix[index].Y, 1)
            dataSet['Cluster'].loc[index] = numClusters
        elif numClusters >= maxClusters:
            #print("At Max")
            dataSet.ix[index].Cluster = clusterToAssign 
            newX = clusterCenters[clusterToAssign][0] + (dataSet.ix[index].X - clusterCenters[clusterToAssign][0])/(clusterCenters[clusterToAssign][2]+1.0) 
            newY = clusterCenters[clusterToAssign][1] + (dataSet.ix[index].Y - clusterCenters[clusterToAssign][1])/(clusterCenters[clusterToAssign][2]+1.0) 
            newSize = clusterCenters[clusterToAssign][2] + 1
            clusterCenters[clusterToAssign] = (newX, newY, newSize)   
            dataSet['Cluster'].loc[index] = clusterToAssign        
    
    return clusterCenters

    
def dist(point1, point2):
    sumsq = 0
    for index in range(0, len(point1) - 1):
        sumsq += math.pow(point1[index] - point2[index],2)
    return math.pow(sumsq,.5)


def sumSquaredError():
    pass


def main():
    print("In Main.")
    prob1()
    #prob2()




if __name__ == "__main__":
    main()

