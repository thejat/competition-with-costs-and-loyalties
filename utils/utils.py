from utils.imports import *

#simple one liner functions
def perc_75(x): return np.percentile(x,75) #Percentile calculations
def perc_25(x): return np.percentile(x,25)

if __name__=='__main__':
	data = np.array(list(range(1,101)))
	print(perc_25(data),perc_75(data))