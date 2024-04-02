import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Map 'Gender' to numerical values
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Rename columns for convenience
df.rename(columns={
    'Age': 'age',
    'Gender': 'gender',
    'CustomerID': 'id',
    'Annual Income (k$)': 'income',
    'Spending Score (1-100)': 'score'
}, inplace=True)

# Apply KMeans clustering
kmeans = cluster.KMeans(n_clusters=5)
kmeans = kmeans.fit(df[['income', 'score']])

# Cluster centers
cluster_centers = kmeans.cluster_centers_

# Assign clusters to data points
df['cluster'] = kmeans.labels_

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='income', y='score', hue='cluster', palette='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='*', label='Cluster Centers')
plt.title('K-Means Clustering of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Elbow Method for finding optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['income', 'score']])
    inertia.append(kmeans.inertia_)

# Plotting the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.show()
