#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa


# In[2]:


data_set = np.genfromtxt("hw08_data_set.csv",delimiter=",")
x=data_set[:,0]
y=data_set[:,1]
x_values=[]
y_values=[]
K=5


# In[3]:


#B MATRIX CALCULATED

N=len(x)
B=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i != j and np.linalg.norm(data_set[i] - data_set[j])<= 1.25:
            B[i][j]=1
            if [data_set[i][0],data_set[j][0]] not in x_values:
                x_values.append([data_set[i][0],data_set[j][0]])
                y_values.append([data_set[i][1],data_set[j][1]])
            
    



# In[4]:


#CONNECTIVITY PLOT 
#THIS PART TAKES A FEW SECONDS TO RUN
plt.figure(figsize=(10,10))
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Connectivity Plot")
for i in range(int(len(x_values))):
    plt.plot(x_values[i],y_values[i],color="grey")


    

plt.plot(x,y,linestyle="none",markersize=10,color="black",marker='o')


# In[5]:


#D MATRIX CALCULATED
D=np.zeros((N,N))

for i in range(N):
    D[i][i]=np.sum(B[i])
    D[i][i]=D[i][i]**-0.5


# In[6]:


#L MATRIX CALCULATED
I= np.eye(N,N)
L=I-np.matmul(D,np.matmul(B,D))


# In[7]:


#EIGEN VALUES AND VECTORS CALCULATED
eigen_values,eigen_vectors=np.linalg.eig(L)


# In[8]:


# N x 2 matrix which contains the indices of the eigen values on the second column 
eigen_mat=np.zeros((N,2))
eigen_mat[:,0]=eigen_values
eigen_mat[:,1]=range(0,N)


# In[9]:


#SORTED EIGEN VALUE MATRIX CREATED
sorted_eigen_mat=eigen_mat[np.argsort(eigen_mat[:,0])]


# In[10]:


#Z MATRIX IS INITIALIZED
Z=[]
for i in range(1,K+1):
    Z.append(eigen_vectors[:,int(sorted_eigen_mat[i][1])])
Z=np.transpose(np.array(Z))


# In[11]:


#INITIAL CENTROIDS CREATED
initial_centroids=[]
initial_centroids.append(Z[28])
initial_centroids.append(Z[142])
initial_centroids.append(Z[203])
initial_centroids.append(Z[270])
initial_centroids.append(Z[276])
initial_centroids=np.array(initial_centroids)


# In[12]:


def update_centroids(memberships, X,inital_centroids):
    K=5
    if memberships is None:
        # initialize centroids
        centroids = initial_centroids
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


# In[13]:


#K-MEANS CLUSTERING
centroids = None
memberships = None

iteration = 1
while True:

    old_centroids = centroids
    centroids = update_centroids(memberships, Z,initial_centroids)
    if np.alltrue(centroids == old_centroids):
        break
    

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break
   
        

    iteration = iteration + 1


# In[14]:


#CLUSTERS ARE SEPERATED
cluster1=[]
cluster2=[]
cluster3=[]
cluster4=[]
cluster5=[]

cluster_list=[cluster1,cluster2,cluster3,cluster4,cluster5]


for i in range(N):
    cluster_list[memberships[i]].append(data_set[i])
for i in range(K):
    cluster_list[i]=np.array(cluster_list[i])


# In[15]:


#CLUSTERS ARE PLOTTED
plt.figure(figsize=(10,10))
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(cluster_list[0][:,0], cluster_list[0][:,1],color="blue")
plt.scatter(cluster_list[1][:,0], cluster_list[1][:,1],color="green")
plt.scatter(cluster_list[2][:,0], cluster_list[2][:,1],color="red")
plt.scatter(cluster_list[3][:,0], cluster_list[3][:,1],color="orange")
plt.scatter(cluster_list[4][:,0], cluster_list[4][:,1],color="purple")

colors=["blue","green","red","orange","purple"]
 
for i in range(K):
    plt.plot(np.mean(cluster_list[i][:,0]),np.mean(cluster_list[i][:,1]),marker="s",markersize=18,color=colors[i],markeredgecolor = "black")    
    
    


# In[ ]:




