#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Load the dataset
df = pd.read_csv("Mall_Customers.csv")  
df.head()

#Basic info and checks
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

#Encode 'Gender' column (optional)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df.head()

#Visualize target features
sns.pairplot(df[['Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

#Select features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Find optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

#Fit K-Means with optimal number of clusters (i.e, k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

#Visualize Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=80)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

#bonus:Analyze average metrics per cluster
df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)

from sklearn.cluster import DBSCAN
#bonus:Try DBSCAN for comparison
db = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = db.fit_predict(X_scaled)

sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Cluster', palette='tab10', s=80)
plt.title('DBSCAN Clustering')
plt.show()



