import matplotlib.pyplot as plt
import numpy as np 

X = np.array([
	[1,2],
	[0.8,1.2],
	[5,8],
	[8,8],
	[1,0.5],
	[9,10]])

colors = 10*['r','g','b','c','y','k']

#all you want to know is if the 2 groups are formed, hence
#you can pass the same data as your testing data.  
class k_means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self,data):
			self.centroids = {}

			#k=2, so just saying that the first 2 centroids are the first 2 elements	
			for i in range(self.k):
				self.centroids[i] = data[i]

			#begin the optimation process
			for i in range(self.max_iter):
				self.classifications = {}

				for i in range(self.k):
					self.classifications[i] = []

				for featureset in data:
					distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]	
					classification = distances.index(min(distances))
					self.classifications[classification].append(featureset)

				prev_centroids = dict(self.centroids)
				
				for classification in self.classifications:
					self.centroids[classification] = np.average(self.classifications[classification], axis=0)

				optimized = True

				for c in self.centroids:
					original_centroid = prev_centroids[c]
					current_centroid = self.centroids[c]
					if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
						optimized = False
				
				if optimized:
					break					
							
	def predict(self,data):
			distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
			classification = distance.index(min(distances))
			return classification		

clf = k_means()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
		marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()		

				