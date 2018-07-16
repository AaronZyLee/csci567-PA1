from typing import List

import numpy as np
import math

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    sum = 0
    for i in range(0,len(y_true)):
        sum += (y_true[i]-y_pred[i])**2
    return sum/len(y_true)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    mat = np.zeros((2,2))
    for i in range(0,len(real_labels)):
        if(predicted_labels[i]==1):
            if(real_labels[i]==1):
                mat[1,1]+=1
            else:
                mat[1,0]+=1
        else:
            if(real_labels[i]==1):
                mat[0,1]+=1
            else:
                mat[0,0]+=1
    precision = mat[1][1]/(mat[1][0]+mat[1][1])
    recall = mat[1][1]/(mat[1][1]+mat[0][1])
    if mat[1][1] == 0:
        return 0
    return 2*precision*recall/(precision+recall)


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    feature = np.array(features)
    poly = np.array(features)
    for i in range(2,k+1):
        poly = np.column_stack((poly,feature**i))
    return poly.tolist()


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    dis=0;
    for i in range(0,len(point1)):
        dis+=(point1[i]-point2[i])**2
    return math.sqrt(dis)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    dis=0
    for i in range(0,len(point1)):
        dis+=(point1[i]*point2[i])
    return dis


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    dis=0;
    for i in range(0,len(point1)):
        dis+=(point1[i]-point2[i])**2
    return -math.exp(-0.5*dis)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        length = len(features)
        zeros = np.zeros(len(features[0]))
        features = np.array(features)       
        for i in range(0,length):
            if (features[i]==zeros).all():
                continue
            div = math.sqrt((features[i]*features[i]).sum())             
            features[i] = features[i]/div
        return features.tolist()


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
    	1. Use a variable to check for first __call__ and only compute
			and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
           	is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    first_call = True
    
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        features = np.array(features)
        if self.first_call == True:
            self.min = features.min(axis=0)
            self.max = features.max(axis=0)
            self.range = self.max-self.min
            self.first_call = False
        features = (features-self.min)/self.range
        return features   
            
            
            
            
