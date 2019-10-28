# -*- coding: utf-8 -*-
"""
A KNN regressor

Instructions: 
    - insert your code where you see the comment '# your code here'
    - you are allowed to add additional methods to the KNNRegressor class
    - do not touch any code outside the KNNRegressor class except to
      add imports
    - obviously, do not use the Scikit-learn KNN classes!

@author: 
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
# you may add imports here

class KNNRegressor:
    """ a KNN regressor """
    
    def __init__(self, k):
        """ Create a knn classifier with the given value k"""
        
        # your code here
        
    def fit(self, X, y):
        """ X is a 2D numeric array and y is a 1D array """
        
        # your code here

    
    def predict(self, X):
        """ Return an array containing the predicted class for each row of X """

        # your code here
    
    def score(self, X, y):
        """ 
        Return the R^2 value for the given data.
        See documentation on score() method for 
        sklearn.neighbors.KNeighborsRegressor to see how
        to calculate R^2.
        """

        # your code here


def main():
    """ Tests """
        
    # data
            
    df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)    
    df['Private'] = (df['Private'] == 'Yes').astype(int)
    
    X = df[['Expend', 'Grad.Rate']].apply(zscore).values
    y = df['Outstate'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # test this class
    
    knn = KNNRegressor(3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    rmse = np.sqrt(((predictions - y_test)**2).mean())
    print('my code: rmse: {:.4f}, R^2: {:.4f}'.format(rmse, knn.score(X_test, y_test)))
    
    # compare to scikit learn
    
    from sklearn.neighbors import KNeighborsRegressor
    
    knn1 = KNeighborsRegressor(3)
    knn1.fit(X_train, y_train)
    predictions1 = knn1.predict(X_test)
    rmse1 = np.sqrt(((predictions1 - y_test)**2).mean())
    print('sklearn: rmse: {:.4f}, R^2: {:.4f}'.format(rmse1, knn1.score(X_test, y_test)))
    
if __name__ == "__main__":
    main()


