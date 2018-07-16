from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

import operator

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.neighbors = features
        self.val = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        result=[]
        for i in range(0,len(features)):
            disArr = []
            for j in range(0,len(self.neighbors)):
                disArr.append(self.distance_function(features[i],self.neighbors[j]))
            disArr = numpy.array(disArr)
            indexArr = disArr.argsort()
            classCount = {}
            for k in range(0,self.k+1):
                voteLabel = self.val[indexArr[k]]
                classCount[voteLabel]=classCount.get(voteLabel,0)+1
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            result.append(sortedClassCount[0][0])
        return result


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
