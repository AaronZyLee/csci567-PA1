from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args :
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''

        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias
            labels : label of each feature [-1,1]

            Returns :
                True/ False : return True if the algorithm converges else False.
        '''
        ############################################################################
        # TODO : complete this function.
        # This should take a list of features and labels [-1,1] and should update
        # to correct weights w. Note that w[0] is the bias term. and first term is
        # expected to be 1 --- accounting for the bias
        ############################################################################
        converge = False
        features = np.array(features)
        iter=0
        while iter<self.max_iteration and not converge:
            iter+=1
            i = 0
            while i<len(features):
                temp = (self.w*features[i]).sum()
                if temp*labels[i]<=self.margin:
                    self.w += labels[i]*features[i]/np.linalg.norm(features[i])
                    break
                i+=1
            if i==len(features):
                converge = True
        return converge

    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias

            Returns :
                labels : List of integers of [-1,1]
        '''
        ############################################################################
        # TODO : complete this function.
        # This should take a list of features and labels [-1,1] and use the learned
        # weights to predict the label
        ############################################################################
        features = np.array(features)
        prediction = (features*self.w).sum(axis=1)
        for i in range(0,len(prediction)):
            if prediction[i]>0:
                prediction[i] = 1
            else:
                prediction[i] = -1
        return prediction.tolist()

    def get_weights(self) -> List[float]:
        return self.w

