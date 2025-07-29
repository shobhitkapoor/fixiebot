import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/historical_tickets.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Customer_Description"])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

kmeans = KMeans(n_clusters=4)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.title("Fix Clustering View")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("../data/fixie_clusters.png")
print("✅ Cluster visualization saved")
