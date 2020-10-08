


import numpy as np
import powerlaw
import numpy as np
from scipy.optimize import curve_fit



'''
given that we know which type of trajectory it is, 
we can calculate alpha:
CTRW:
    CTRW with waiting times distribution  
    ψ(t) ∼ t^1−σ and the step variance √ D and mean zero 
	gives alpha = σ - 1
	
	we estimate it using Dist_distribution(x,y) function

Levy walks 
	time between steps, are retrieved from ψ(t) ∼ t^(-1-σ)
	
ATTM
	we have different parts of trajectories with diffusion coefficients D_i for τi, 
	τi = D−γi , the anomalous exponent is shown to be α = σ/γ, 
	we need to get two parameters:
	1. gamma - from distribution of time changes when diffusion coefficients changes
	we estimate it using function ???
	2. alpha - from distribution of time changes between consequent time-steps 
	we estimate it using function Dist_distribution

	
	
'''

import numpy as np
import seaborn
from matplotlib import pyplot as plt


'''
input: 

data1 is an trajectory, we get array of sequences X_1(t), X_2(t),... X_n(t), 
where n is number of dimensions, t is time. 

'''

#size = 100
#data1 = np.random.random((size, size)) # for example we take random sample

lags = range(2,100)
def hurst_exponen_data(p):
    '''
    given series p(t), where t is time 
    p(t) is format of zip(list) of arrays from X and Y
    '''    
    variancetau = []; tau = []

    for lag in lags: 
        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)
        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    hurst = m[0] / 2

    return hurst




def func_fit(x, a):
    return  x^^(a) #a * np.exp(-b * x) + c


def Dist_distribution(x_data, y_data):
    '''
    This function calculates the distribution of distance travelled per unit of time and retuns an histogram of this distribution. Note that we fixed the range and number of bins for simplicity, but this might need to be 
    '''
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2 + (y[:,:-1]-y[:,1:])**2) for x,y in zip(x_data, y_data)]
    max_value = [np.max(i) for i in step_dist]
    distributions = [np.histogram(sd,range=[0,ix],bins=20)[0] for sd,ix in zip(step_dist,max_value)]
    l_dist = [distributions[i].shape[0] for i in np.arange(len(distributions))]
    #return distributions, l_dist

	return step_dist 
	
	
def Time_distribution(x_data, y_data):
    '''
    This function calculates the distribution of times between consequent jumps
	'''
    # 1. calculate distribution between points
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2 + (y[:,:-1]-y[:,1:])**2) for x,y in zip(x_data, y_data)]

    # 2. if distance between points is 0 then RW does not move 
	print(step_dist)
	print(np.nonzero(step_dist))
	nnz_steps = np.nonzero(step_dist)
	
	# 3. then time distribution is difference between each two non-zero arrays
	time_dist = np.diff(nnz_steps)
    #max_value = [np.max(i) for i in step_dist]
    #distributions = [np.histogram(sd,range=[0,ix],bins=20)[0] for sd,ix in zip(step_dist,max_value)]
    #l_dist = [distributions[i].shape[0] for i in np.arange(len(distributions))]
    #return distributions, l_dist

	return time_dist 
	
	

def Convex_hull(x_data,y_data, delta):
    traj = [np.concatenate((i,j)).T for i,j in zip(x_data,y_data)]
    vol_tot=[]
    for k in np.arange(len(traj)):
        vol=[]
        for i in np.arange(len(traj[k])-delta-1):
            try:
                hul = ConvexHull(traj[k][i:i+delta,:])
                vol.append(hul.volume)
            except:
                vol.append(0)
        vol_tot.append(np.array(vol).reshape((1,-1))) 
    l_dist = [i.shape[1] for i in vol_tot]
	
	time_dist = threshold(l_dist) # times when convex hull is changing from one regime to another 
	
    return time_dist #vol_tot, l_dist, 

def alpha_calculation(x, y, cat):

	''' 
	ATTM does not work yet 
	CTRW does work 
	FBM needs to be checked with https://pypi.org/project/fbm/ hurst exponent 
	SBM 
	'''

    if cat == 0: # ATTM
        # msd = (x-x[:,0])**2 + (y-y[:,0])**2
		
		# Difdist is distribution of diffusion coefficients for trajectory
		#delta = 5
		#Difdist = Convex_hull_regime(x_data,y_data, delta)
		#gamma  =  powerlaw.Fit(Difdist.reshape(-1)).power_law.alpha
		#psi = Dist_distribution(x,y)
        #sigma = powerlaw.Fit(psi.reshape(-1)).power_law.alpha
        #alpha = sigma * 1./gamma #powerlaw.Fit(msd.reshape(-1)).power_law.alpha
		
		psi = Dist_distribution(x,y)
		Dist_distribution(x,y)
        alpha = powerlaw.Fit(psi.reshape(-1)).power_law.alpha - 1 
		
        
    if cat == 1: # CTRW
		# estimate waiting time steps distribution psi(t)
        #msd = (x-x[:,0])**2 + (y-y[:,0])**2
        #powerlaw.Fit(msd.reshape(-1)).power_law.alpha

		psi = Dist_distribution(x,y)
        alpha = powerlaw.Fit(psi.reshape(-1)).power_law.alpha - 1 # alpha = sigma -1 

        
    if cat == 2: # FBM
		p = zip(x,y)
		hurst_exp = hurst_exponen_data(p)
	
        msd = (x-x[:,0])**2 + (y-y[:,0])**2
        alpha = powerlaw.Fit(msd.reshape(-1)).power_law.alpha
        
    if cat == 3: # LW
        msd = (x-x[:,0])**2 + (y-y[:,0])**2
		sigma = Time_distribution(x_data, y_data) 
		if (sigma >0)&(sigma<1): 
			alpha = 2
		if (sigma>1)&(sigma<2)	
		#alpha = powerlaw.Fit(msd.reshape(-1)).power_law.alpha
        
    if cat == 3: # SBM
        msd = (x-x[:,0])**2 + (y-y[:,0])**2
        alpha = powerlaw.Fit(msd.reshape(-1)).power_law.alpha
		
		return alpha 