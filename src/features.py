import numpy as np
from scipy.spatial import ConvexHull


'''
1D
'''

def Q_measure_1D(x_data, tau):
    '''
    This function calculates the convex hull measure for an ensemble of trajctories. The input has to be a list of x and y coordinates arrays as well as the \tau value. This function returns the convex hull time-series and the length of each series  
    '''
    Q = [(x[:,:-tau]+x[:,tau:])**2 for x in x_data]
    Q = [k/np.max(k) for k in Q]
    length_Q = [np.shape(a)[1] for a in Q]
    return Q, length_Q



def Dist_distribution_1D(x_data):
    '''
    This function calculates the distribution of distance travelled per unit of time and retuns an histogram of this distribution. Note that we fixed the range and number of bins for simplicity, but this might need to be 
    '''
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2) for x in x_data]
    max_value = [np.max(i) for i in step_dist]
    distributions = [np.histogram(sd,range=[0,ix],bins=20, density=True)[0] for sd,ix in zip(step_dist,max_value)]
    l_dist = [distributions[i].shape[0] for i in np.arange(len(distributions))]
    return distributions, l_dist

'''
2D
'''

def Q_measure_2D(x_data, y_data , tau):
    '''
    This function calculates the convex hull measure for an ensemble of trajctories. The input has to be a list of x and y coordinates arrays as well as the \tau value. This function returns the convex hull time-series and the length of each series  
    '''
    Q = [(x[:,:-tau]+x[:,tau:])**2 + (y[:,:-tau]+y[:,tau:])**2 for x,y in zip(x_data, y_data)]
    Q = [k/np.max(k) for k in Q]
    length_Q = [np.shape(a)[1] for a in Q]
    return Q, length_Q


def Dist_distribution(x_data, y_data):
    '''
    This function calculates the distribution of distance travelled per unit of time and retuns an histogram of this distribution. Note that we fixed the range and number of bins for simplicity, but this might need to be 
    '''
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2 + (y[:,:-1]-y[:,1:])**2) for x,y in zip(x_data, y_data)]
    max_value = [np.max(i) for i in step_dist]
    distributions = [np.histogram(sd,range=[0,ix],bins=20)[0] for sd,ix in zip(step_dist,max_value)]
    l_dist = [distributions[i].shape[0] for i in np.arange(len(distributions))]
    return distributions, l_dist

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
    return vol_tot, l_dist