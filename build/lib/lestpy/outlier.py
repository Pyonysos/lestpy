# list of imports


class Outliers_Inspection:
    '''
    Outliers_Inspection:
    
        arguments:
            other: Object

        methods:
            
            - cooks_distance()
                
                params:
                    self
                    plot : Boolean, default True. If True, the 
                
                return:
            
            - mahalanobis_distance()
            
                D**2 = (x-µ)**T.C**(-1).(x-µ)
                where, 
                - D**2        is the square of the Mahalanobis distance. 
                - x          is the vector of the observation (row in a dataset), 
                - µ          is the vector of mean values of independent variables (mean of each column), 
                - C**(-1)     is the inverse covariance matrix of independent variables.
                
                params:
                    plot
                
                return:
                    self.mahal_d.diagonal():

            - z_score: calculate the z-score of the targets.
                    z-score = (y - mean(y)) / std(y)
                    z-score is the number of standard deviations away from the mean the data point is.
                
                params:
                    ddof: degre of freedom
                
                returns:
                    DataFrame
    '''

    def __init__(self, other:object):
        self.other=other
        self.frame_list=[]
        for i in self.other.y:
            self.outliers = outliers_influence.OLSInfluence(self.other.model[i]['model_final'])
            self.frame_list.append(self.outliers.summary_frame())
        return self
        
    def cooks_distance(self, plot: bool =True):
        for i in self.other.y:
            threshold = 4/self.outliers.summary_frame().shape[0]
            
            print(f'threshold (4/n) = {round(threshold,3)}' )
            outliers_list= []
            for n in range(0,self.outliers.summary_frame().shape[0]):
                if self.outliers.summary_frame().at[n, 'cooks_d'] >= threshold:
                    outliers_list.append((n, self.outliers.summary_frame().at[n, 'cooks_d']))
            print(f'potential outliers : {outliers_list}')
            if plot:
                #plt.figure()
                plt.scatter(range(0,self.outliers.summary_frame().shape[0]), self.outliers.summary_frame()['cooks_d'], label=i)
            print(self.outliers.summary_table())
            
        plt.plot([0, self.outliers.summary_frame().shape[0]], [threshold, threshold], c='r', label='threshold')
        plt.xlabel('Observation indices')
        plt.ylabel('Cook\'s distance')
        plt.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left', borderaxespad=0.)
        plt.show()
        
        return self.frame_list
    
    def mahalanobis_distance(self, plot:bool=True):
      #To Do
        """
        mahalanobis_distance:
        D**2 = (x-µ)**T.C**(-1).(x-µ)
        where, 
        - D**2        is the square of the Mahalanobis distance. 
        - x          is the vector of the observation (row in a dataset), 
        - µ          is the vector of mean values of independent variables (mean of each column), 
        - C**(-1)     is the inverse covariance matrix of independent variables.
        """
        for i in self.orner.y:
            diff_x_u = self.other.X - np.mean(self.other.X, axis=0)
            if not cov:
                cov = np.cov(self.other.X.values.T)
            inv_covmat = sp.linalg.inv(cov)
            left_term = np.dot(diff_x_u, inv_covmat)
            self.mahal_d = np.dot(left_term, diff_x_u.T)
        #ajouter Mahalanobis a outlier summary
        
        if plot:
          print("Not yet implemented")
          plt.scatter(range(0,self.outliers.summary_frame().shape[0]), self.mahal_d.diagonal(), label=i)
        
        return self.mahal_d.diagonal()

    def z_score(self, ddof:int=0, plot:bool=True)-> DataFrame:
        """
        https://medium.com/clarusway/z-score-and-how-its-used-to-determine-an-outlier-642110f3b482
        
        z_score: calculate the z-score of the targets.
            z-score = (y - mean(y)) / std(y)
        z-score is the number of standard deviations away from the mean the data point is.
        
        params:
            ddof: degre of freedom
        returns:
            DataFrame
        """
        df = pd.DataFrame()
        for col in self.other.y.columns:
            col_zscore = col + "_zscore"
            df[col_zscore] = (self.other.y[col] - self.other.y[col].mean())/self.other.y[col].std(ddof=ddof)
        """
        df["outlier"] = (abs(df[col + "_zscore"])>3).astype(int)
        print("number of outliers = " + str(df.outlier.value_counts()[1]))
        """
        return df

"""
    def z_score(df, col, min_z=1, max_z = 5, step = 0.1, print_list = False):
        z_scores = df["Data_zscore"]
        threshold_list = []
        for threshold in np.arange(min_z, max_z, step):
            threshold_list.append((threshold, len(np.where(z_scores > threshold)[0])))
            df_outlier = pd.DataFrame(threshold_list, columns = ['threshold', 'outlier_count'])
            df_outlier['pct'] = (df_outlier.outlier_count - df_outlier.outlier_count.shift(-1))/df_outlier.outlier_count*100
        plt.plot(df_outlier.threshold, df_outlier.outlier_count)
        best_treshold = round(df_outlier.iloc[df_outlier.pct.argmax(), 0],2)
        outlier_limit = int(df[col].dropna().mean() + (df[col].dropna().std()) * df_outlier.iloc[df_outlier.pct.argmax(), 0])
        percentile_threshold = stats.percentileofscore(df[col].dropna(), outlier_limit)
        plt.vlines(best_treshold, 0, df_outlier.outlier_count.max(), 
               colors="r", ls = ":"
              )
        plt.annotate("Zscore : {}\nValue : {}\nPercentile : {}".format(best_treshold, outlier_limit,(np.round(percentile_threshold, 3),np.round(100-percentile_threshold, 3))),(best_treshold, df_outlier.outlier_count.max()/2))
        #plt.show()
        if print_list:
            print(df_outlier)
        return (plt, df_outlier, best_treshold, outlier_limit, percentile_threshold)
"""
"""
    def outlier_inspect(df, col, min_z=1, max_z = 5, step = 0.2, max_hist = None, bins = 50):
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(col, fontsize=16)
        plt.subplot(1,3,1)
        if max_hist == None:
            sns.histplot(df[col], kde=False, bins = 50,color="r")
        else :
            sns.distplot(df[df[col]<=max_hist][col], kde=False, bins = 50)
        plt.subplot(1,3,2)
        sns.boxplot(df[col])
        plt.subplot(1,3,3)
        z_score_inspect = z_score(df, col, min_z=min_z, max_z = max_z, step = step)
        plt.show()
"""