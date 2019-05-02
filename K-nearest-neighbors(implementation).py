from math import sqrt
import numpy as np 
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('you are an idiot, set a proper k value')

	distances = []
	for group in data:
		for features in data[group]:
			euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidian_distance, group])

	#gives you the first three closest distances
	votes = [i[1] for i in sorted(distances)[:k]]
	#using a counter to keep track which group is on top
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k	
    	
	return vote_result,confidence


df = pd.read_csv('breast-cancer-wisconsin.data')	
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

#there are some values in the df that are strings
#making every thing a float
full_data = df.astype(float).values.tolist()

#shuffle the data
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
#first 80%
train_data = full_data[:-int(test_size*len(full_data))]
#last 20%
test_data = full_data[-int(test_size*len(full_data)): ]

for i in train_data:
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	test_set[i[-1]].append(i[:-1])	

correct = 0 
total = 0

for group in train_set:
	for data in test_set[group]:
		#scyitlearn uses a default k value of 5
		vote,confidence = k_nearest_neighbors(train_set, data, k=5)
		if group == vote:
			correct += 1	
		total += 1

print('accuracy:', correct/total)	
print(confidence)	

#accuracy comes from testing the classifire against different data points(using testing data)
#confidence comes from the number of votes that are in favour of the final descission
#if all the votes are for one class 100% confident
#if 4 ofut of the five votes are for 1 class then 80% confidence
			
#you can thread k_nearest_neigbours for super fast accuracy
# n_jobs=-1 for maximum threading

#k_nearest_neighbours can be used for both linear and non-linear data. 

