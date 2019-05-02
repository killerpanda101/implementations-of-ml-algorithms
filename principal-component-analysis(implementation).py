import matplotlib.pyplot as plt 
import numpy as np 
from numpy import linalg as LA

import pandas as pd
df=pd.read_csv('breast-cancer-wisconsin.data', sep=',',header=None)

#normalizing the data
max = []
min = []
for col in range(df.shape[1]):
	max.append(df[col].max()) 
	min.append(df[col].min()) 
	
for col in range(df.shape[1]):
	df[col] = (df[col] - min[col])/(max[col]-min[col])	

#finding the corelation matrix
c = np.cov(df.T)
#finding the eign vectors and eign values
eig_val, eig_vec = LA.eig(c)

# Make a list of (eigenvalue, eigenvector) tuples
              #absolute eig value  #the corresponding eig vector for that value   
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]


# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

#taking onnly the top 2 eign values
matrix_w = np.hstack((eig_pairs[0][1].reshape(11,1), 
                      eig_pairs[1][1].reshape(11,1)))

#reducing the number of columns in main matrix(pca)
Y = df.dot(matrix_w)

#plotting the data
plt.scatter(Y[0],Y[1], label='PCA', color='b')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis')
plt.show()




			


    	
    	
    	
    	
	    	
    	


    	
    	


       

			
	


	


