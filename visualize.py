from igraph import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#define plot style to render consistant figures between one another
class PlotStyle():
    def __init__(self, projet_name, cmap, figsize, pov):
        '''
        project_name: string, all figures will be saved with a name starting with project_name value
        cmap: string, name of the matplotlib colormap to be used throughout the figures
        pov: tuple for azimut and elevation for '3d projection'
        '''
        pass


def graph(features, responses=None, render= adj_matrix, plot=True):
    '''
    graph:
    
    params:
        dataset: DataFrame
        responses: DataFrame
        render:
        plot: True

    return: None
    '''
    result=None

    return result



def plot_surface(a,b,c, **kwargs):
    """
    plot_surface: plot response surface in function of features a and b
    
    params:
    a: 1D array (n,)
    b: 1D array (n,)
    c: 1D array (n,)
    kwargs
    
    return: None
    """
    if hasattr(kwargs, "cmap"):
        cmap=kwargs['cmap']
    else:
        cmap='Viridis'
    
    fig = plt.figure(figsize=(40,40))
    ax = plt.axes(projection='3d')
    Cmap = plt.get_cmap(cmap)  
    surf = ax.plot_trisurf(a.ravel(), b.ravel(), c, cmap=Cmap, antialiased=True, edgecolor='none')     
    fig.colorbar(surf, ax =ax, shrink=0.5, aspect=5)
    ax.set_xlabel(f'{a.name}')
    ax.set_ylabel(f'{b.name}')
    ax.set_zlabel(f'{c.name}')
    ax.set_title(f'Interaction: {c.name}')
    fig.show()



def pareto_frontier(dataset, objectives: list, target: list = ['maximize', 'maximize'], plot: bool = True):

        """
        pareto_frontier: Extract from dataset the undominated datas according to objectives
        
        params:
            dataset : pandas DataFrame of the data to analyze
            objectives: list, names of the two features in the dataset to analyze. 
            target: list
                'maximize' : the feature should be maximized
                'minimize' : the feature should be minimized
            plot: boolean, 
                True: the pareto_frontier will be plotted using matplolib 

        return:
            p_frontX: list of the undominated datas of objectives[0] according to the target criteria
            p_frontY: list of the undominated datas of objectives[1] according to the target criteria


        according to: Jamie Bull | jamiebull1@gmail.com
        https://oco-carbon.com/metrics/find-pareto-frontiers-in-python/
        """
        # Sort the list in either ascending or descending order of X
        if target[0]=='maximize':
            maxX = True
        elif target[0]=='minimize':
            maxX= False

        maxY = False if target[1] == 'minimize' else True
        
        if len(objectives) > 2:
            raise ValueError('Length of "objectives" should be 2.')

        myList = sorted([[dataset[objectives[0]][i], dataset[objectives[1]][i]] for i in range(len(dataset[objectives[0]]))], reverse=maxX)
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]    
        # Loop through the sorted list
        for pair in myList[1:]:
            if maxY: 
                if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
            else:
                if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
        # Turn resulting pairs back into a list of Xs and Ys
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]

        mask = dataset[objectives[0]].isin(p_frontX)

        if plot:
            plt.scatter(dataset[objectives[0]], dataset[objectives[1]], alpha=0.5, c='lightgrey', label='trials')
            plt.scatter(dataset[objectives[0]].mask(~mask), dataset[objectives[1]].mask(~mask), label='undominated trials')
            # Then plot the Pareto frontier on top
            plt.plot(p_frontX, p_frontY, c='r', label='pareto front')
            plt.xlabel(dataset[objectives[0]].name)
            plt.ylabel(dataset[objectives[1]].name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()

        return p_frontX, p_frontY