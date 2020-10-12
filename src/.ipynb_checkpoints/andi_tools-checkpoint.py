import numpy as np
import pandas as pd

def Andi_to_xy_testdata(data):
    
    '''
    This function takes as input the 2D Andi data set and outputs the corresponding x,y data lists
    '''
    
    T_lens = [np.int((k.shape[1]-2)/2)for k in data]
    x_data, y_data = [k[:,2:T_lens[i]+2] for  i, k in enumerate(data)], [k[:,T_lens[i]+2:] for i, k in enumerate(data)]
    return x_data, y_data


def Andi_to_xy(path_input, path_labels):
    
    '''
    This function takes as input the 2D Andi data set and outputs the corresponding x,y data lists
    '''
    all_2D =  pd.read_csv(path_input, index_col=0).values[:,1:]
    data = [all_2D[i][~np.isnan(all_2D[i])].reshape(1,-1) for i in np.arange(len(all_2D))]
    T_lens = [np.int((k.shape[1]-3)/2)for k in data]
    xy_data = [np.concatenate([k[:,3:T_lens[i]+3],k[:,T_lens[i]+3:]], axis=0) for  i, k in enumerate(data)]
    
    # Extract the labels
    labels = pd.read_csv(path_labels).values[:,1]
    output = np.array([np.eye(5)[labels[i]] for i in range(len(labels))])
    return xy_data, output


def group_by_length(feature, feature_range):
    
    '''
    This function groups a list of trajectories of variable length into grouped arrays of identical length
    '''
    
    grouped_list=[]
    for j in feature_range:
        grouped_list.append(np.array([x for i,x in enumerate(feature) if np.shape(x)[1] == j]))
    return grouped_list

def group_similar_as(feature, feature_map, feature_range):
    
    '''
    This function groups a list of features of variable length into grouped arrays of identical length, following another feature. This is usefull to ensure that all features are grouped identically. 
    '''
    
    grouped_list=[]
    for j in feature_range:
        grouped_list.append(np.array([feature[i] for i,x in enumerate(feature_map) if np.shape(x)[1] == j]))
    return grouped_list
