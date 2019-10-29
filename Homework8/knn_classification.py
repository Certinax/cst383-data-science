# -*- coding: utf-8 -*-
"""
A KNN classifier

Instructions: 
    - insert your code where you see the comment '# your code here'
    - you are allowed to add additional methods to the KNNClassify class
    - do not touch any code outside the KNNClassify class except to
      add imports
    - obviously, do not use the Scikit-learn KNN classes!

@author: 
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import zscore, mode
from sklearn.model_selection import train_test_split
# you may add imports here

class KNNClassify:
    """ a KNN classifier """
    
    def __init__(self, k):
        """ Create a knn classifier with the given value k"""
        self.k = k
        # your code here
        
    def fit(self, X, y):
        """ X is a 2D numeric array and y is a 1D array """
        self.X = X
        self.y = y
        # your code here
        
    def modeMe(self, x):
        return mode(x)[0][0]
    
    def predict(self, X):
        """ Return an array containing the predicted class for each row of X """
        # your code here
        dm = distance_matrix(X, self.X)
        smallest_values = np.argsort(dm)[:,:self.k]
        closest_values = self.y[smallest_values]
            
        nearest = np.apply_along_axis(self.modeMe, 1, closest_values)
        
        return nearest

    def numOfCorrectPredictions(self, predictions, y_test):
        correct_predict = 0
        
        for i, j in enumerate(y_test):
            if(y_test[i] == predictions[i]):
                correct_predict += 1
                
        return correct_predict


    def score(self, X, y):
        """ Return the accuracy of this classifier on the given data """
        # your code here
        predictions = self.predict(X)
        
        correct_predictions = self.numOfCorrectPredictions(predictions, y)
        
        result = correct_predictions/y.size
        
        return result

def main():
    """ Tests """
        
    # set up the data
    
    df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)    
    df['Private'] = (df['Private'] == 'Yes').astype(int)

    X = df[['F.Undergrad', 'Top10perc']].apply(zscore).values
    y = df['Private'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # test this class
    
    knn = KNNClassify(5)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print('my accuracy: {:.4f}'.format(accuracy))
    
    # compare to scikit learn
    
    from sklearn.neighbors import KNeighborsClassifier
    
    knn1 = KNeighborsClassifier(5)
    knn1.fit(X_train, y_train)
    accuracy1 = knn1.score(X_test, y_test)
    print('sklearn accuracy: {:.4f}'.format(accuracy1))
    
if __name__ == "__main__":
    main()




