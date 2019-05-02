from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
import random

#how many entries we want, step is by how much you want them to vary
def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if correlation == 'pos':
			val += step
		elif correlation == 'neg':
			val -= step		
	xs = [i for i in range(hm)]		
     
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
 
xs,ys = create_dataset(40,10,2,correlation='pos')

def best_fit_line(xs,ys):
	m = ((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs)) 
	b = mean(ys) - m*mean(xs)
	return(m,b)

m,b = best_fit_line(xs,ys)

#list of ys based on the regression line
regression_line = [(m*x) + b for x in xs]

#difference between input y and the y on the line
def squared_error(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)

def coff_of_determination(ys_orig,ys_line):
	#an list where every value is mean of ys.
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regre = squared_error(ys_orig,ys_line)
	squared_error_mean = squared_error(ys_orig,y_mean_line)
	return(1-(squared_error_regre/squared_error_mean))

r_squared = coff_of_determination(ys, regression_line)
print(r_squared)