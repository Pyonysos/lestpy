#from igraph import *



"""
    def pareto_frontier(res, objectives: list, target: list = ['maximize', 'maximize'], plot: bool = True):
        
        '''
        according to: Jamie Bull | jamiebull1@gmail.com
        https://oco-carbon.com/metrics/find-pareto-frontiers-in-python/
        '''

        # Sort the list in either ascending or descending order of X
        if target[0]=='maximize':
            maxX = True
        elif target[0]=='minimize':
            maxX= False

        maxY = False if target[1] == 'minimize' else True
        
        if len(objectives) > 2:
            raise ValueError('Length of "objectives" should be 2.')

        myList = sorted([[res[objectives[0]][i], res[objectives[1]][i]] for i in range(len(res[objectives[0]]))], reverse=maxX)
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

        mask = res[objectives[0]].isin(p_frontX)

        if plot:
            plt.scatter(res[objectives[0]], res[objectives[1]], alpha=0.5, c='lightgrey', label='trials')
            plt.scatter(res[objectives[0]].mask(~mask), res[objectives[1]].mask(~mask), label='undominated trials')
            # Then plot the Pareto frontier on top
            plt.plot(p_frontX, p_frontY, c='r', label='pareto front')
            plt.xlabel(res[objectives[0]].name)
            plt.ylabel(res[objectives[1]].name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.show()

        return p_frontX, p_frontY
"""