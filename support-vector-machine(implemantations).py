import mathplotlib.pyplot as pyt 
from mathplotlib import style
import numpy as np 
style.use('ggplot')

class Support_vector_Machine:
	
	#like public static void main
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1:'r', -1:'b'} #dictionary
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)

	#training
	def fit(self, data):
	    self.data =  data
	    # { ||W|| : [w,b]}
	    opt_dict = {}
	    transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]
        #finding the ranges (min and the max)
	    all_data = []
	    for yi in self.data:
	    	for featureset in self.data[yi]:
	    		all_data.append(feature)
	    		#these are the max and min for features values
	    		#-1 min and 8 max
	    self.max_feature_value = max(all_data)
	    self.min_feature_value = min(all_data)
	    #FINDING THE STEP SIZES
	    # support vector yi(xi+b) = 1, you go as close as 1.001
	    step_size = [self.max_feature_value * 0.1,
	    			 self.max_feature_value * 0.01,
	    			 # point of expence 
	    			 self.max_feature_value * 0.001]

	    #extreamly expensive
	    #you dont have to take such small steps for b as it not that valuable
	    b_range_multiple = 5
	    # step multiple
	    b_multiple = 5
	    #max feature value value * 10
	    latest_optimum = self.max_feature_value*10

	    
		#starting the stepping process
		for step in step_size:
			#replace w 8*10, helps speed up the calculation
			w = np.array([latest_optimum,latest_optimum])
			#we can do this as only one minimum(convex optimization problem)
			optimized = False
			while not optimized: 
				# we take 5 times large steps than what we do with w
				#you dont need such a accurate value for b
				#Arange also lets you specify my how much you want to increment each time
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
								   self.max_feature_value*b_range_multiple
								   , step*b_multiple)
					for transformation in transforms:
						w_t = w*transformation #multiplying with 1,1 -1,1 1,-1 -1,1
						found_option = True
						#weakest link in all of svm
						#smo tries to fix this a bit
						for i in self.data: # i is -1,1
							for xi in self.data[i]: # the class values -1 or 1
								yi=i
								if not yi*(np.dot(w_t,xi)+b) >= 1:
									found_option = False
								
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]

				if w[0] < 0:
					optimized True
					print('Optimized a step.')
				else:
					# w-[step,step]
					w = w - step

		#just a sorted list of all the magnitudes
		norms = sorted([n for n in opt_dict])
		#the optimal choice is the first entry of the sorted norms
		opt_choice = norms[0]
		# ||W|| = [w,b]
		self.w = opt_choice[0]
		self.b = opt_choice[1]
		latest_optimum = opt_choice[0][0]+step*2				

			
		

	def predict(self, features):
		# sign( x.w+b )
		classification = np.sign(np.dot(np.array(features),self.w)+self.b)    	
		if classification !=0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors(classification))
		return classification
		

	def visualize(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

		#hyperplane = x.w+b
		#V=x.W+b
		#positive support vector =1, negetive support vector =-1, descision boundary =0
		def hyperplane(x,w,b,v):
			return(-w[0]*x-b+v)/w[1]
			
		datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)	
		hyp_x_min = datarange[0] 	
		hyp_x_max = datarange[1]

		# (w.x+b) = 1 positive support vector hyperplane
		#these are the y values for a given value of x
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) #this is just a scaler value
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

		# (w.x+b) = -1 negetive support vector hyperplane
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) 
		nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

		# (w.x+b) = 0 descision boundary
		db1 = hyperplane(hyp_x_min, self.w, self.b, 0) 
		db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

		plt.show()


             #keys    #values
data_dict = {-1:np.array([[1,7],
	                      [2,8],
	                      [3,8]]),

	         1:np.array([[5,1],
	         	         [6,-1],
	         	         [7,3]])}


svm =Support_vector_Machine()
svm.fit(data = data_dict)
predict_us = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]
for p in predict_us:
	svm.predict(p)
svm.visualize()	         

			
			
			