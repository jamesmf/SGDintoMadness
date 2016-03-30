# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:21:47 2016

@author: test
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D #Uncomment this to get d3 js plots
#import mpld3 #Uncomment this to get d3 js plots
import numpy as np
import csv

"""**************************************************************"""
#generate 1D, 2D, 3D plots to demonstrate curse of dimensionality
data3d  =  [[1,2,5],
            [2,4,4],
            [3,9,1],
            [4,3,2],
            [5,7,6],
            [6,6,8],
            [7,1,9],
            [8,8,3],
            [9,5,7]]
            
data3d  = np.array(data3d)

fig     = plt.figure(0)
ax      = fig.add_subplot(111,projection='3d')
ax.scatter(data3d[:,0],data3d[:,1],data3d[:,2])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")

plt.savefig("as3d")


#2D
fig2    = plt.figure(1)
ax2      = fig2.add_subplot(111)
ax2.scatter(list(data3d[:,0]),list(data3d[:,1]))
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
#html2d  = mpld3.fig_to_html(fig2)
#with open("html2d.txt",'wb') as f:
#    f.write(html2d)
plt.savefig("as2d")


#3D
fig3    = plt.figure(2)
ax3     = fig3.add_subplot(111)
ax3.scatter(data3d[:,0],np.ones(data3d.shape[0]))
ax3.set_xlabel("x1")
ax3.get_yaxis().set_visible(False)

html1d  = mpld3.fig_to_html(fig3)
#with open("html1d.txt",'wb') as f:
#    f.write(html1d)
plt.savefig("as1d")


"""**************************************************************"""
#Demonstrate PCA
data    =  [[1,75],
            [2,80],
            [3,79],
            [4,84],
            [5,91],
            [6,87],
            [7,99],
            [8,89],
            [9,97]]
data    = np.array(data)
#PCA requires us to center the data by subtracting the mean
#PCA object will take care of this for us if we don't
mean    = np.mean(data,axis=0) 

            
fig5    = plt.figure(4)
ax      = fig5.add_subplot(111)
ax.set_xlim([0,10])
ax.set_ylim([70,100])
ax.scatter(data[:,0],data[:,1])
ax.set_title("Test Data")
ax.set_xlabel("Hours Spent Studying")
ax.set_ylabel("Test Score")
plt.savefig("beforepca")


data    = data - mean
pca     = PCA(n_components=1) #this is a lossy transformation - 2D in, 1D out
transf  = pca.fit_transform(data)


print pca.components_ #this prints the principle components


#Plot the data in PCA space
fig6    = plt.figure(5)
ax      = fig6.add_subplot(111)
pca2    = PCA(n_components=2)
transf  = pca2.fit_transform(data)
ax.scatter(transf[:,0],transf[:,1])
ax.scatter(transf[:,0],np.zeros(transf[:,0].shape),color='r')
ax.set_title("Test Data Projected Into PCA Space")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.savefig("2Dpca")


#Plot the data projected from the 1D first principle component back to original 2D space
fig7    = plt.figure(6)
ax      = fig7.add_subplot(111)
ax.set_ylim([70,100])
new     = pca.transform(data)
new     = pca.inverse_transform(new)
new     = new+mean
ax.scatter(new[:,0],new[:,1],color='r')
data    = data+mean
ax.scatter(data[:,0],data[:,1],color='b')
ax.plot(new[:,0],new[:,1],color='k')
ax.set_title("Test Data Recovered from Principal Component 1")
ax.set_xlabel("Hours Spent Studying")
ax.set_ylabel("Test Score")
plt.savefig("backfrom1Dpca")



"""**************************************************************"""
#Demonstrate k-means clustering
data1   = np.random.normal(loc=-0.75,scale=0.2,size=(50,2)) #distribution 1
data2   = np.random.rand(150,2)*3-1.5 #distribution 2
data3   = np.random.normal(loc=0.75,scale=0.2,size=(30,2)) #distribution 3
data    = np.append(np.append(data1,data2,axis=0),data3,axis=0) #combine them
fig8    = plt.figure(7)
ax      = fig8.add_subplot(111)
ax.scatter(data[:,0],data[:,1]) #plot data
plt.savefig("dataPreCluster")


#show several different clustering for values of k
ks  = [2,4,20]

for run,k in enumerate(ks):
    km  = KMeans(n_clusters=k)  #new KMeans object
    km.fit(data)                #fit it to the data
    rain    = cm.rainbow(np.linspace(0,1,k)) #assign each cluster index a matplotlib color
    clrs    = [rain[i] for i in km.labels_] #transform label indices into colors
    
    fig     = plt.figure(8+run)     #new figure
    ax      = fig.add_subplot(111)  
    ax.scatter(data[:,0],data[:,1],c=clrs)  #plot the data coloring by cluster
    plt.savefig("kmeans_"+str(k))



#show the change in clustering "score" as k increases
ks  = range(2,100,4)
scores  = []
for run,k in enumerate(ks):
    km  = KMeans(n_clusters=k)
    km.fit(data)
    scores.append(-km.score(data))
    
fig     = plt.figure(20)
ax      = fig.add_subplot(111)
ax.scatter(ks,scores)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Sum of Distances to Centroids")
plt.savefig("kmeansScores")

