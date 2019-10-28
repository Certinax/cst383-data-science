# -*- coding: utf-8 -*-
"""
A class to predict college tuition.

Instructions:
    
Modify the code below only where you see the comment '# your code here'.
Be sure not to modify the code in main().

Use the Scikit-Learn KNeighborsRegressor class to predict college tuition.
Try to find the columns of the data, and the value of k,
that gives the best results.

Mostly you will just need to call methods of the KNeighborsRegressor
class.

Make sure to save the indexes of the columns you want to use,
and use those columns to select the columns of X when you call
the KNeighborsRegressor methods.

Do not modify any code 

"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class TuitionPredictor:
    """ Predict college tuitions using KNN regression.  """
    
    def __init__(self, X, y):
        """ 
        X_train is a dataframe of all columns from the College data set except
        for the college name and 'Outstate'.  The data is scaled and all columns 
        are numeric.  (Column 'private' is included, but as a numeric column.)
        y_train is a corresponding array of 'Outstate' values from the College
        data set.
        """

        # your code here
        # your code will need to:
        # - save the indexes of the columns you want to use
        # - decide on value of k that you want to use
        # - create, train, and save a KNeighborsRegressor object
        
    def predict(self, X):
        """
        Return the predicted tuition for each row of X.
        X is a dataframe of the same type as X_train in method __init__.
        """
        
        # your code here
        # you will use your KNeighborsRegressor object
    
    def score(self, X, y):
        """
        Return the R^2 score for for the given test data.
        """
        
        # your code here
        # you will use your KNeighborsRegressor here
    
    def rmse(self, X, y):
        """
        Return the room mean squared error score for for the given test data.
        """
        
        # your code here
        # you will use your KNeighborsRegressor to make predictions, and
        # then compute the RMSE


def main():
    """ Tests """
        
    import pandas as pd
    from scipy.stats import zscore
    from sklearn.model_selection import train_test_split 
            
    df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)    
    df['Private'] = (df['Private'] == 'Yes').astype(int)
    
    y = df['Outstate'].values
    df.drop('Outstate', inplace=True, axis=1)
    X = df.apply(zscore).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    tp = TuitionPredictor(X_train, y_train)
    
    print('score: {:.4f}'.format(tp.score(X_test, y_test)))
    print('rmse: {:.4f}'.format(tp.rmse(X_test, y_test)))
    
if __name__ == "__main__":
    main()



