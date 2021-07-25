"""
**************************************************************************************************************************
DEPENDENCIES
**************************************************************************************************************************
"""


from pandas.core.frame import DataFrame
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d

from scipy.stats import dirichlet

import random

import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import pandas as pd
import numpy as np

import time


interaction_dict={}

"""
**************************************************************************************************************************
INTERACTION CLASSES 
**************************************************************************************************************************
"""
class Interaction:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        if not hasattr(self.x, 'name') or not hasattr(self.y, 'name'):
            self.x = pd.DataFrame(self.x, name='x')
            self.y = pd.DataFrame(self.y, name='y')
     
    def compute(self):
        new_var = pd.DataFrame(self.calc(), columns=[self.name])
        return new_var   

class X_fort_Quand_Y_faible_Et_Inversement(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort quand {self.y.name} faible et inversement'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(np.multiply(-self.x, self.y)).reshape(-1,1)
        return func
        
class X_fort_Ou_Y_fort(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort ou {self.y.name} fort'
        self.interaction = self.__class__.__name__       
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(-(np.max(self.x)-self.x)*(np.max(self.y)-self.y)).reshape(-1,1)
        return func
    
class X_fort_Ou_Y_faible(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort ou {self.y.name} faible'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
    
    def calc(self):
        func = np.array(-(np.max(self.x)-self.x)*(np.abs(np.min(self.y))+self.y)).reshape(-1,1)
        return func

class X_et_Y_forts(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} et {self.y.name} forts'
        self.interaction = self.__class__.__name__ 
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array((self.x+np.abs(np.min(self.x)))*(self.y+np.abs(np.min(self.y)))).reshape(-1,1)
        return func
                                        
class X_fort_et_Y_faible(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort et {self.y.name} faible'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array((self.x+np.abs(np.min(self.x)))*(np.max(self.y)-self.y)).reshape(-1,1)
        return func

class X_fort_si_Y_fort(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort si {self.y.name} fort'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(self.x*(np.abs(np.min(self.y))+self.y)).reshape(-1,1)
        return func
                                          
class X_fort_si_Y_faible(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort si {self.y.name} faible'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(self.x*(np.abs(np.max(self.y))-self.y)).reshape(-1,1)
        return func
         
class X_fort_si_Y_moyen(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} fort si {self.y.name} moyen'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
    
    def calc(self):
        func = np.array(self.x / np.sqrt((np.max(self.y)+np.abs(np.min(self.y)))/500+np.square(self.y))).reshape(-1,1)
        return func
        
class X_moyen_si_Y_fort(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} moyen si {self.y.name} fort'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array((np.abs(np.min(self.y))+self.y)/np.sqrt((np.max(self.x)+np.abs(np.min(self.x)))/200+np.square(self.x))).reshape(-1,1)
        return func

class X_moyen_si_Y_faible(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} moyen si {self.y.name} faible'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array((np.max(self.y)-self.y)/np.sqrt((np.max(self.x)+np.abs(np.min(self.x)))/200+np.square(self.x))).reshape(-1,1)
        return func
    
class Ni_X_ni_Y_extremes(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'Ni {self.x.name} ni {self.y.name} extremes'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(-np.square(self.x)-np.square(self.y)).reshape(-1,1)
        return func
    
class X_Y_moyens(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} et {self.y.name} moyens'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array((np.square(np.max(self.x))-np.square(self.x))*(np.square(np.max(self.y))-np.square(self.y))).reshape(-1,1)
        return func
    
class X_comme_Y(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'{self.x.name} comme {self.y.name}'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(-np.square(self.x-self.y)).reshape(-1,1)
        return func
    
class Somme_X_et_Y_forte(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'Somme {self.x.name} et {self.y.name} forte'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(self.x+self.y).reshape(-1,1)
        return func
    
class Difference_X_et_Y_forte(Interaction):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.name = f'Difference {self.x.name} et {self.y.name} forte'
        self.interaction = self.__class__.__name__
        interaction_dict[self.name] = {'x' : self.x, 'y' : self.y, 'interaction' : self.interaction}
        
    def calc(self):
        func = np.array(self.x-self.y).reshape(-1,1)
        return func

"""
**************************************************************************************************************************
CORICO ALGORHITHM
**************************************************************************************************************************
"""

class LBM_Regression:
    def __init__(self):
        return
    
    def bibliography(self):
        print("""
                Bibliography :
                1. Lesty, Michel, et P Buat-Ménard. "La synthèse géométrique des corrélations multidimensionnelles". Les Cahiers de l’Analyse des données VII, no 3 (1982): 355‑70.
                2. Lesty, Michel. "Une nouvelle approche dans le choix des régresseurs de la régression multiple en présence d’interactions et de colinéarités". revue Modulad 22 (1999): 41‑77.
                3. Derringer, George and Suich, Ronald. "Simultaneous Optimization of Several Response Variables". Journal of Quality Technology 12 (1980): 214-219. 
                """)    
    
    def __autointeraction_param(self, allow_autointeraction):        
        return 1 if allow_autointeraction==True else 0
    
    def __compute_interaction(self, X, autointeraction, interaction_list):
        new_X = X.reset_index(drop=True)
        for i in range(X.shape[1]):
            for j in range(i+1-autointeraction, X.shape[1]):
                for interaction in interaction_list:
                    if i != j:
                        #changer en numpy
                        new_X = pd.concat((new_X, interaction(X.iloc[:,i], X.iloc[:,j]).compute()), axis =1)
                    else:
                        if not interaction in [Difference_X_et_Y_forte, X_comme_Y]:
                            new_X = pd.concat((new_X, interaction(X.iloc[:,i], X.iloc[:,j]).compute()), axis =1)
        
        self.with_interactions = True
                            
        return new_X
    
    def __rescale_data(self, X, y, scaler):
        
        #normalisation des données
        if scaler =='robust':
            self.transformer = RobustScaler()
        elif scaler =='minmax':
            self.transformer = MinMaxScaler()
        elif scaler =='standard':
            self.transformer = StandardScaler()
        
        return self.transformer.fit_transform(X, y)
    
    def __variable_instant(self, X):
        """
        transformation of the matrix into a modified unit called "variable/instant"
        """
        eye = np.eye(X.shape[0], X.shape[0])

        X_sqr = np.sum(np.multiply(X, X), axis=0)
        X_sum = np.square(np.sum(X, axis=0))
        self.denomin = np.sqrt(X.shape[0]*X_sqr-X_sum)*np.sqrt(X.shape[0]-1)

        self.Coef = np.sum(eye, axis=0).reshape(-1,1).dot(np.sum(np.array(X), axis=0).reshape(1,-1))[0]
        self.Coef = np.array(self.Coef).reshape(1, self.Coef.shape[0])
        
        self.with_variable_instant = True
        self.Shape = X.shape[0]
        
        variable = np.divide(np.subtract(X*self.Shape, self.Coef), self.denomin)

        return variable
    
    def __unscale_data(self, rescaled_X):
        """
        inverse transformation of __variable_instant 
        """
        
        if self.with_variable_instant:
            unscaled_X = (rescaled_X*self.denomin+self.Coef)/self.Shape
        else: 
            unscaled_X = rescaled_X
        
        return unscaled_X
    
    def __compute_correlation_matrix(self, X, y):
        if hasattr(y, 'name'):
            name = y.name
        else :
            name=y.columns
        
        self.corr_X = pd.DataFrame(np.corrcoef(pd.concat([X, y], axis=1).T), columns= X.columns.tolist() + [name])
        return
        
    def __feature_selection(self, res, corr_X, M, res_list, threshold):
    
        #identifier la meilleure variable explicative
        corr_X= np.abs(corr_X)
        
        best_interaction = corr_X.iloc[-1, : -1].idxmax(axis=1)
        best_corr = M.loc[:, best_interaction]
        best_coef = corr_X.iloc[-1, : -1].max()

        if best_coef >= threshold : 
            #enregistrement du nom de la correlation et du coef dans la liste de reponses
            res_list.append([best_interaction, round(best_coef, 3), corr_X.columns.get_loc(best_interaction)])

            #Ajout de la colonne à la matrice des résultats
            res = pd.concat((res, best_corr), axis=1)

        return res, res_list
        
    
    def __partial_correlations(self, correlation_matrix, interaction):
    
        """
        Partial correlation Formula :
        rAB.C = rAB-rAC.rBC/(sqrt(1-rAC^2).sqrt(1-rBC^2)) 
        """

        reg = correlation_matrix.iloc[:, correlation_matrix.columns.get_loc(interaction)]

        rAC = np.array(reg).reshape(reg.shape[0], 1)
        rBC = np.array(reg).reshape(1, reg.shape[0])

        matrice = np.array(correlation_matrix)

        numerator = np.subtract(matrice, rAC.dot(rBC))

        #avoid negative values in sqrt
        rAC_sqr = np.square(rAC)
        rAC_sqr[rAC_sqr > 1]=0.99
        rBC_sqr = np.square(rBC)
        rBC_sqr[rBC_sqr > 1]=0.99

        denominator = np.sqrt(np.subtract(np.ones(rAC.shape),rAC_sqr)).dot(np.sqrt(np.subtract(np.ones(rBC.shape),rBC_sqr)))

        #avoid zero division
        denominator[denominator==0] = 1000


        correlation_matrix = pd.DataFrame(np.divide(numerator,denominator), columns=correlation_matrix.columns)

        for i,j in range(correlation_matrix.shape[0], correlation_matrix.shape[0]):
            correlation_matrix.iloc[i,j] = 1

        return correlation_matrix
    
    
    def __model_evaluation(self, mat_res):
    
        model = LinearRegression()

        #Calcul de Q2 global
        X = np.array(mat_res.iloc[:,1:])
        y = np.array(mat_res.iloc[:,0])
        
        loot=LeaveOneOut()
        loot.get_n_splits(X)

        SSres = []
        SStot = []

        #Leave-one-out cross validation
        for train_index, test_index in loot.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                SSres.append((float(y_test[0])-float(y_pred[0]))**2)
                SStot.append(float(y_test[0]))


        SStot = np.array(SStot)
        SStot_mean = np.multiply(np.ones(SStot.shape), np.mean(SStot))    

        return round(1-np.sum(SSres)/np.sum(np.square(SStot-SStot_mean)),3)
    
    
    def __desirability_DS(self, target, target_weights, prediction):
        """
        Computation of desirability according to Derringer and Suich (1980) for multiple response optimization
        
        parameters:
            target : String, float, int or List of those
                    if string -> 'maximize' or 'minimize' or 'none'
                    float or int should correspond to the values that are targeted during optimization
                    List in the order of the columns in the dataFrame of responses
                    2 types of values are accepted : 
                                - String ('maximize' or 'minimize')
                                - Float (or int) if the optimization must target a specific value
        return
            DataFrame
            
        """
       
        desirability = pd.DataFrame(np.ones((prediction.shape[0], 1)), columns=['desirability']) 
                 
        for i in range(0, prediction.shape[1]):
            if prediction.shape[1] > 1:
                location = prediction.iloc[:, i]
            else:
                location = prediction
        
            if type(target) is str or type(target) is float or type(target) is int or type(target) is None:
                goal = target
            elif type(target) is list:
                goal= target[i]
            else:
                raise TypeError("target is either a string, a float, an integer or a list")

                    
            if goal=='maximize': #desirability to maximise the response
                objective = np.divide(location - location.min(axis = 0), location.max(axis=0)- location.min(axis=0))
            elif goal=='minimize': #desirability to minimize the response
                objective = np.divide(location.max(axis = 0) - location, location.max(axis=0) - location.min(axis=0))
            elif (goal == 'none') or (goal is None): #desirability to reach a specific target value
                objective = 0
            else:
                Solution1 = (location - location.min(axis=0))/ (goal - location.min(axis=0))
                Solution2 = (location - location.max(axis=0))/ (goal - location.max(axis=0))
                objective = np.minimum(Solution1, Solution2)
                    
            if target_weights is None:
                target_weights = [1 for n in range(0,prediction.shape[1])]
                    

            desirability = np.multiply(desirability, np.power(np.array(objective), target_weights[i]/np.sum(target_weights)).reshape(-1,1))
        return desirability
    
    def transform(self, X, y=None, scaler: str ='robust', variable_instant:bool=True, allow_autointeraction=False, 
                  interaction_list: list =[X_fort_Quand_Y_faible_Et_Inversement, X_fort_Ou_Y_fort, X_fort_Ou_Y_faible, 
                                    X_et_Y_forts,X_fort_et_Y_faible, X_fort_si_Y_fort, X_fort_si_Y_faible, 
                                    X_fort_si_Y_moyen, X_moyen_si_Y_fort, Ni_X_ni_Y_extremes, X_Y_moyens, 
                                    X_comme_Y, Somme_X_et_Y_forte, Difference_X_et_Y_forte]):
        """
        transform method :
        
        Params:
            X : DataFrame, matrix of the features
            y : DataFrame, matrix of the targer
            scaler : string, correspond to the method that will be used to rescale the data before computation of the interactions
            variable_instant : boolean, if True, data and computed interactions will be rescaled according to the "variable-instant" method of Lesty et al. (1999)
            allow_autointeraction : boolean, if True, additional interactions of the features with themselves will be considered
            interaction_list : list, List of interactions that are found relevant to the studied problem
        
        Return :
            self
        
        """
        if scaler not in ['robust', 'standard', 'minmax']:
            raise ValueError(f'{scaler} method is not implemented,\n  Implemented methods "robust, minmax and standard"')
        
        #retourne tableau de données avec interactions
        start = time.time()
        
        self.X = X.reset_index(drop=True)
        self.X_start = self.X
        self.y = y.reset_index(drop=True)
        
        #Step1: set autointeraction
        try:
            autointeraction = self.__autointeraction_param(allow_autointeraction)
        except:
            raise NotImplementedError('autointeraction failed')
       
        #Step2: Rescale data
        try:
            self.X = pd.DataFrame(self.__rescale_data(self.X, self.y, scaler), columns = self.X.columns)
        except:
            raise NotImplementedError('rescaling data failed')
    
        #Step2: compute new features
        try:
            self.features = self.__compute_interaction(self.X, autointeraction, interaction_list)
        except: 
            raise NotImplementedError('computation of the interactions failed')
        
        #Step3: Rescale data
        try:
            if variable_instant:
                try:
                    self.rescaled_features = pd.DataFrame(self.__variable_instant(self.features), columns = self.features.columns)
                    print('method = variable instant')
                except:
                    raise NotImplementedError('rescaling variable_instant data failed')
            else:
                try:
                    self.X = pd.DataFrame(self.__rescale_data(self.X, self.y, scaler), columns = self.X.columns)
                    print(f'method = {scaler}')
                except:
                    raise NotImplementedError(f'rescaling interactions with {scaler} method failed')
        except:
            raise NotImplementedError('rescaling data failed')
            
        end = time.time()
        print(f'calculé en {round(end-start, 3)} secondes')
        
        self.with_transform = True
        
        return self
    

    def fit(self, max_regressors_nb: int = 10, threshold: float = 0.2):
        
        """
        fit method :
        
        Params:
            max_regressors_nb : integer, set the maximum number of features that the model can use to describe the target
            threshold : float, threshold of partial correlation under which the model with stop including features into the model
        
        Return :
            self
        
        """
        
        #Mesure du temps de calcul
        start = time.time()
        
        if type(threshold) is not float:
            raise TypeError('threshold must be a float between 0 and 1')
        
        if type(max_regressors_nb) is not int:
            raise TypeError('max_regressors_nb must be an integer')
        
        self.model={}
        
        try:
            y = self.y.to_frame()
        except:
            y = self.y
        
        for i in y:
            self.model[i] = {}
            self.model[i]['selected_features'] = []
            self.model[i]['results'] = y[i]
            self.model[i]['metrics'] = []
            
            self.__compute_correlation_matrix(self.rescaled_features,  y[i])
            
            for reg in range(max_regressors_nb):
                #identification de la meilleure interaction
                self.model[i]['results'], self.model[i]['selected_features'] = self.__feature_selection(self.model[i]['results'], self.corr_X, self.rescaled_features, self.model[i]['selected_features'], threshold)
                self.corr_X = self.__partial_correlations(self.corr_X, self.model[i]['selected_features'][-1][0])

                self.model[i]['metrics'].append(self.__model_evaluation(self.model[i]['results']))
                
            self.model[i]['nb_predictor'] = self.model[i]['metrics'].index(max(self.model[i]['metrics']))+1
            self.model[i]['model_final'] = LinearRegression()
                
            self.model[i]['model_final'].fit(self.model[i]['results'].iloc[:,1:self.model[i]['nb_predictor']+1],  y[i])
            self.model[i]['y_pred'] = self.model[i]['model_final'].predict(self.model[i]['results'].iloc[:,1:self.model[i]['nb_predictor']+1])
                
            coefficients = np.array(self.model[i]['model_final'].coef_).reshape(len(self.model[i]['model_final'].coef_), 1)
            print(f'MODELISATION OF THE TARGET {y[i].name}', '\n' , 
                  f'{y[i].name} is high if :', '\n', 
                  pd.concat((pd.DataFrame(coefficients.round(3), columns= ['coefficients']), 
                             pd.DataFrame(np.array(self.model[i]['selected_features'])[:, :-1], columns=['interactions', 'corrCoef_'])), axis=1).dropna(axis=0), 
                  f'model intercept : {self.model[i]["model_final"].intercept_.round(3)}', "\n", 
                  sep="\n")
            end = time.time()
            print(f'fit method computed in {round(end-start, 3)} seconds')
        self.with_fit = True
        return self
    
    def predict(self, X): 
        #transformation of the matrix of parameters to predict
        X = X[self.X_start.columns]
        transformed_X = pd.DataFrame(self.transformer.transform(X), columns=X.columns.tolist())
        
        
        self.y_pred = pd.DataFrame()
        
        try:
            y = self.y.to_frame()
        except:
            y = self.y
        
        for i in y:
            new_X = None
        
        #computation of the selected and enginered features of the model
            for element in self.model[i]['selected_features'][:self.model[i]['nb_predictor']]:
                try :
                    func = interaction_dict[element[0]]['interaction']
                    x_df= transformed_X[interaction_dict[element[0]]["x"].name]
                    y_df= transformed_X[interaction_dict[element[0]]["y"].name]
                    col = eval(func)(x_df, y_df).compute()
                except KeyError:
                    col = transformed_X[element[0]]
            
                finally:
                    pd.options.mode.use_inf_as_na = True
                    
                    if any(col.isna()):
                        col[col.isna()] = 0
                    
                    

                    if 'new_X' not in locals():
                        new_X = col
                    else:
                        new_X = pd.concat((new_X, col), axis=1)
            
            #transformation to acurate scale
            L = [int(n) for n in np.array(self.model[i]['selected_features'][:self.model[i]['nb_predictor']])[:,-1]]
            Coef = self.Coef[:,L]

            Denomin = pd.DataFrame(self.denomin).iloc[L].transpose()
            Denominator = np.array(Denomin).reshape(1, Denomin.shape[1])

            var_X = np.divide(np.subtract(np.array(new_X)*self.Shape, Coef), Denominator)
            variable_instant_X = pd.DataFrame(var_X, columns=Denomin.columns.tolist())

            
            #computation with the coefficients
            self.model[i]['y_pred'] = pd.DataFrame(np.dot(variable_instant_X, self.model[i]['model_final'].coef_.reshape(-1,1)) + np.array(self.model[i]['model_final'].intercept_).reshape(1,1), columns=[f'Pred{i}'])
            
            self.y_pred = pd.concat((self.y_pred, self.model[i]['y_pred']), axis=1)
        
        return self.y_pred
    
    def features_analysis(self, X):
        self.experimental_domain = {}
        
        for feature in X.columns.tolist():
            #print(set([int(num) for num in X[feature].unique()]))
            if len(X[feature].unique()) == len(set([int(num) for num in X[feature].values])):
            #(len(X[feature].unique()) < 6 ) or ((len(X[feature].unique()) / X.shape[0] ) < 0.05) or X[feature].dtype is int :
                vartype = 'discrete'
                varlist = X[feature].unique().tolist()
            else:
                vartype = 'continuous'
                varlist = None
            
            self.experimental_domain[X[feature].name] = [None, X[feature].min(axis=0), X[feature].max(axis=0), varlist , vartype]
        
        self.mix=None
        for i in range(0, X.shape[1]):
            for j in range(i+1, X.shape[1]+1):
                a = X.iloc[:, i:j].sum(axis=1)
                
                if sum(np.abs(a - a.mean())) < sum(np.abs(a*0.05)):
                    self.mix=X.iloc[:, i:j].columns.tolist()
                    self.mixmax = X.iloc[:, i:j].max(axis=0).mean()
                    self.mixmin = X.iloc[:, i:j].min(axis=0).mean()
                    break
        
        exp_dom = pd.DataFrame(self.experimental_domain, index=['status','min value', 'max_value', 'values', 'var type'])
        print('experimental domain: ', exp_dom, 'mixture: ', self.mix, sep='\n\n' )

        return self.experimental_domain, self.mix

    def __features_generator(self, remaining_features, experimental_domain, size):
        #create random array respecting the restrictions of the features 
        for var in remaining_features:
            if experimental_domain[var][4] == 'discrete':
                #create a random array of the discrete values
                exploration_array = np.random.choice(experimental_domain[var][2], (size, 1), replace=True, p=None) #p peut permettre de mettre du poids sur le paramètre interessant
            elif experimental_domain[var][4] == 'continuous':
                #create a random array of the discrete values
                rng = np.random.default_rng()
                exploration_array = (experimental_domain[var][2] - experimental_domain[var][1]) * rng.random((size, 1), dtype=np.float64) + experimental_domain[var][1]
            try:
                experimental_domain[var][5] = exploration_array
            except:
                experimental_domain[var].append(exploration_array)  
        return experimental_domain
    
    def __mix_features_generator(self, alpha, size, random_state, mix):
        if alpha is None:
             alpha = np.ones((len(mix))) / len(mix)
        return pd.DataFrame(dirichlet.rvs(alpha, size=size, random_state=random_state), columns = mix)
    
    def generator(self, experimental_domain, mix, alpha : list, size: int, random_state: int=None):
        x=None
        
        if mix is not None:
                x = self.mixmax * self.__mix_features_generator(alpha, size, random_state, mix) - self.mixmin
                
                remaining_features = list(set(self.X.columns.tolist()) - set(mix))
        else:
            remaining_features = experimental_domain.keys()
        print("size = " + str(size))   
        if len(remaining_features) > 0:
            experimental_domain = self.__features_generator(remaining_features, experimental_domain, size)
            for var in remaining_features:
                if x is None:
                    x = pd.DataFrame()   
                #a voir si erreur
                x = pd.concat((x, pd.DataFrame(experimental_domain[var][5], columns=[var])), axis=1)
        return x
    
    def optimize(self, experimental_domain:dict=None, target:list=None, target_weights:list=None, mix:list = None, alpha : list=None, size: int= 10000, random_state: int=None):
        
        #etude de la qualité des paramètres (quantitatif ou qualitatif)
        if experimental_domain is None:
            experimental_domain, mix = self.features_analysis(self.X_start)
              
        elif type(experimental_domain) is not dict:
            raise TypeError('experimental_domain must be a dictionary')
        
        #generation d'un nouveau tableau exploratoire
        sample = self.generator(experimental_domain, mix, alpha, size, random_state)

        #calcul des targets
        if self.with_transform:
            prediction = self.predict(sample)

            desirability = self.__desirability_DS(target, target_weights, prediction)

                #rendu
            res = pd.concat((sample, prediction, desirability), axis=1)
            print("Mean of the 5 best results", np.around(res.nlargest(5, 'desirability').mean(axis=0), 3), "Best result", np.around(res.iloc[res['desirability'].idxmax(axis=1), :], 3), sep='\n\n')
            print('Sample description:', sample.describe().round(2), sep='\n\n')
            
            return res   
    
    def fitting_score(self, y):
        self.model[y.name]['r2score'] = r2_score(y, self.model[y.name]['y_pred']).round(3)
        self.model[y.name]['adjustedr2score'] = (1-(1-self.model[y.name]['r2score'])*(self.model[y.name]['results'].shape[0]-1)/(self.model[y.name]['results'].shape[0]-self.model[y.name]['results'].shape[1]-1)).round(3)
        self.model[y.name]['Q2_obs'] = self.model[y.name]['metrics'][self.model[y.name]['nb_predictor']-1]
        stats = pd.DataFrame(np.array([self.model[y.name]['r2score'], self.model[y.name]['adjustedr2score'], self.model[y.name]['Q2_obs'].round(3)]).reshape(1,-1), columns=['R²', 'adj-R²','calc-Q²'], index = ['model score'])
        print(stats)
        return
    
    def score(self, X, y):
        return
    
    
   
    def residues_hist(self, y):
        plt.hist(y-self.model[y.name]['y_pred'], bins=10)
        plt.xlabel('residual distance')
        plt.ylabel('observation number')
        plt.title('Distribution of the residuals')
        return
    
    def fit_scatter(self, y):
        plt.scatter(y, self.model[y.name]['y_pred'])
        plt.plot([np.min(y), np.max(y)], [np.min(y),np.max(y)], c='r')
        plt.xlabel(f'measured values of {self.model[y.name]["results"].iloc[:,0].name}')
        plt.ylabel(f'predicted values of {self.model[y.name]["results"].iloc[:,0].name}')
        plt.title('Measured vs. predicted values')
        return
    
    def metrics_curve(self, y):
        plt.plot(np.arange(1, len(self.model[y.name]['metrics'])+1), self.model[y.name]['metrics'])
        plt.plot(self.model[y.name]['nb_predictor'], self.model[y.name]['metrics'][self.model[y.name]['nb_predictor']-1], linestyle='none', marker='o', label='Optimal number of predictors')
        plt.legend()
        plt.xlabel('number of predictors')
        plt.ylabel('Q² values')
        plt.title('Evolution of Q² with increasing number of predictors')
        return
    
    def describe(self, X, y):
        #retourne score, histogramme des résidus et diagonale
        try:
            y = self.y.to_frame()
        except:
            y = self.y
        
        for i in y:
            print(i)
            self.fitting_score(y[i])
            plt.figure(figsize=(15,10))
            plt.subplot(2, 2, 1)
            self.fit_scatter(y[i])
            plt.subplot(2, 2, 3)
            self.residues_hist(y[i])
            plt.subplot(2, 2, 4)
            self.metrics_curve(y[i])
            plt.suptitle('Overview of the modelisation')
        return
    
    def print_in_file():
        return #fichier avec données enregistrées et formatées 

    def __extract_features(self, experimental_domain: dict):

        try:
            screened_var = [key for key in experimental_domain.keys() if experimental_domain[key][0] == 'toPlot']

            return screened_var

        except ValueError:
            print('To plot a ternary diagram please set 3 variable values to None')
        
    def __generate_ternary_matrix(self, experimental_domain, mix, alpha, size, random_state):
        #list the features to plot
        var = self.__extract_features(experimental_domain)
        #generate a dataset
        Arr = self.__mix_features_generator(alpha, size, random_state, var)
        
        #scale the data to the right values
        df_Arr = (self.mixmax * pd.DataFrame(Arr, columns=var) - self.mixmin)

        #list the feature not plot
        set_values = []
        set_values_names = []

        for key, value in experimental_domain.items():
            if isinstance(experimental_domain[key][0], (int, float)):
                set_values.append(value[0])
                set_values_names.append(key)
            #if experimental_domain[key][0] is None:
                
                

        #broadcast the set values to complete the dataset and
        Bc_set_values = np.broadcast_to(np.array(set_values).reshape(1,-1), (size, len(set_values)))
        X = pd.DataFrame(np.hstack((df_Arr, Bc_set_values)), columns = var + set_values_names)

        """
        <!!> Probleme si toutes les variables ne font pas partie d'un plan de mélange !!!
        """
        #set the mixture to the acurate sum
        #print(X)
        X_mix = 100* X[mix] / np.sum(X[mix], axis=1).mean()
        X = pd.concat((X[set(X.columns)-set(X_mix.columns)], X_mix), axis=1)
        #print(X[self.X_start.columns])
        Results = self.predict(X[self.X_start.columns])

        return var, X, Results

    def TM_plot(self, experimental_domain: dict, mix: list = None, alpha: list=None, size: int = 1000, random_state: int=None, ncontours: int=20):
        #generate the accurate data to be plot
        if mix is None:
            if self.with_fit:
                mix = self.mix
        var, X, results = self.__generate_ternary_matrix(experimental_domain, mix, alpha, size, random_state)
        
        #plot the ternary contour for each targets
        for res in results:
            fig = ff.create_ternary_contour(np.array(X[var]).T, np.array(results[res]).T, pole_labels=var, interp_mode='cartesian', colorscale='Viridis', showscale=True, ncontours=ncontours)
            fig.show()
        return self
    
    def RS_plot(self, X: DataFrame = None, Y: DataFrame = None, experimental_domain: dict=None, size: int = 10000):
        #If variables are undefined by user, get the variables that were use for modelling
        if self.with_fit:
            if X is None:
                X = self.X_start 
            if Y is None:
                Y= self.y
            if experimental_domain is None:
                experimental_domain = self.experimental_domain
        else : 
            #if fit method has not been used yet -> user must define X and Y
            if (X is None) or (Y is None):
                raise ValueError('')
        
        #List the variables to plot in the model
        features_to_plot = self.__extract_features(experimental_domain)
        
        #generate the data that will be plotted
        X_complete = self.generator(experimental_domain= experimental_domain, mix= None, alpha= None, size=size)

        #Set the fixed variables to the desired value
        Arr=[]
        Arr_name=[]
        for key in (set(experimental_domain.keys())-set(features_to_plot)):
            if not isinstance(experimental_domain[key][0], (int, float)):
                Arr.append(0)
            else:
                Arr.append(experimental_domain[key][0])
            Arr_name.append(key)
        X_complete[Arr_name] = pd.DataFrame(np.full((size, len(Arr)), Arr), columns = Arr_name)
        print(X_complete.describe())
        
        #Compute the output values of the data
        Y = self.predict(X_complete[self.X_start.columns])

        #Plot the surface for each target
        a,b = X_complete.iloc[:, 0], X_complete.iloc[:,1]
        for c in Y:

            fig = plt.figure(figsize=(14,9))    
            ax = plt.axes(projection='3d')
            Cmap = plt.get_cmap('viridis')

            Z = Y[c].values.tolist()            
            surf = ax.plot_trisurf(a.values.tolist(), b.values.tolist(), Z, cmap=Cmap, antialiased=True, edgecolor='none')
            fig.colorbar(surf, ax =ax, shrink=0.5, aspect=5)
            ax.set_xlabel(f'{a.name}')
            ax.set_ylabel(f'{b.name}')
            ax.set_zlabel(f'{c}')
            ax.set_title(f'{c}=f({a.name, b.name}')
                    
            fig.show()