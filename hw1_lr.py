from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        features = numpy.matrix(features)
        values = numpy.matrix(values).T
        features = numpy.column_stack((numpy.ones((features.shape[0],1)), features))
        self.weights = (features.T*features).I*features.T*values


    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        features = numpy.matrix(features)
        features = numpy.column_stack((numpy.ones((features.shape[0],1)), features))
        prediction = features*self.weights
        return prediction.T.tolist()[0]


    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.T.tolist()[0]



class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        features = numpy.matrix(features)
        values = numpy.matrix(values).T
        features = numpy.column_stack((numpy.ones((features.shape[0],1)), features))
        self.weights = (features.T*features+self.alpha*numpy.identity(features.shape[1])).I*features.T*values

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        features = numpy.matrix(features)
        features = numpy.column_stack((numpy.ones((features.shape[0],1)), features))
        prediction = features*self.weights
        return prediction.T.tolist()[0]

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.T.tolist()[0]


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
