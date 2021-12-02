# # -*- coding: utf-8 -*-
# @Author: xiao yu
# @Date:   2021-5-20
# @Last Modified by:   chenkai shen
# @Last Modified time: 2021-11-10
# SmoteR
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, X, Y, ratio=0.5, k=5, r=2):
        self.instancesize, self.n_attrs = X.shape
        self.X = X
        self.Y = Y
        # self.ratio is the desired percentage of the rare instances, default 0.5
        self.ratio = ratio
        #k is the number of nearest neighbors of a defective module,default 5
        self.k = k
        #r is the power parameter for the minkowski distance metric
        self.r = r

    def over_sampling(self):

        normalinstancesize, rareinstanceX, rareinstanceY = self.refreshData(
            self.X, self.Y)
        rareinstancesize = self.instancesize - normalinstancesize
        if self.ratio < rareinstancesize * 1.0 / (rareinstancesize + normalinstancesize):
            p = 0
        elif self.ratio < 2.0 * rareinstancesize / (2.0 * rareinstancesize + normalinstancesize):
            p = int(((self.ratio - 1) * rareinstancesize + self.ratio * normalinstancesize) / (1 - self.ratio))
            keep = np.random.permutation(rareinstancesize)[:p]
            traininginstancesX = rareinstanceX[keep]
            traininginstancesY = rareinstanceY[keep]
            index = 1
        else:
            p = rareinstancesize
            index = int(((self.ratio - 1) * rareinstancesize + self.ratio * normalinstancesize) / (
                        1 - self.ratio) / rareinstancesize)
            traininginstancesX = rareinstanceX
            traininginstancesY = rareinstanceY

        if (p == 0):
            return self.X, self.Y
        else:
            syntheticX=[]
            syntheticY=[]

            for i in range(p):
                nnarray = self.nearestNeighbors(
                    self.r, self.k, targetPoint=traininginstancesX[i], allPoints=rareinstanceX)
                syntheticiX, syntheticiY= self.populate(traininginstancesX[i], traininginstancesY[i], rareinstanceX, rareinstanceY, nnarray, self.r, index)
                syntheticX.append(syntheticiX)
                syntheticY.append(syntheticiY)
            syntheticX, syntheticY = np.asarray(syntheticX), np.asarray(syntheticY)

            syntheticX = np.reshape(syntheticX, (-1, self.n_attrs))
            syntheticY = syntheticY.flatten()

            X = np.vstack((self.X, syntheticX))
            Y = np.hstack((self.Y, syntheticY))

            return X, Y

    def populate(self, traininginstanceX, traininginstanceY, rareinstancesX, rareinstancesY, nnarray, r, index):
        syntheticX=[]
        syntheticY=[]
        for j in range(index):
            nn = np.random.randint(0, self.k)
            nn = min(nn, len(nnarray) - 1)
            dif = rareinstancesX[nnarray[nn]] - traininginstanceX
            gap = np.random.rand(1, self.n_attrs)
            syntheticinstanceX=traininginstanceX + gap.flatten() * dif
            syntheticX.append(syntheticinstanceX)

            dist1 = (float)((np.sum(abs(syntheticinstanceX - traininginstanceX) ** r)) ** (1 / r))
            dist2 = (float)((np.sum(abs(syntheticinstanceX - rareinstancesX[nnarray[nn]]) ** r)) ** (1 / r))
            if (dist1 + dist2 != 0):
                syntheticinstanceY=(dist1 * rareinstancesY[nnarray[nn]] + dist2 * traininginstanceY) * 1.0 / (dist1 + dist2)
                syntheticY.append(syntheticinstanceY)
            else:
                syntheticinstanceY =traininginstanceY * 1.0
                syntheticY.append(syntheticinstanceY)
        return syntheticX, syntheticY


    def refreshData(self, dataX, dataY):
        bugDataX = []
        bugDataY = []
        count = 0
        dataY = np.matrix(dataY).T
        dataX = np.array(dataX)
        for i in range(len(dataY)):
            if dataY[i] > 0:
                bugDataX.append(dataX[i])
                bugDataY.append(int(dataY[i]))
            else:
                count += 1

        bugDataX = np.array(bugDataX)
        bugDataY = np.array(bugDataY)
        return count, bugDataX, bugDataY

    def nearestNeighbors(self, r, k, targetPoint, allPoints):
        candidate = []
        index = 1 / r
        targetPoint = np.asarray(targetPoint)
        allPoints = np.asarray(allPoints)
        if k>len(allPoints):
            nearestneighbors=[i for i in range(len(allPoints))]
            return nearestneighbors

        else:
            for idx, point in enumerate(allPoints):
                subtraction = abs(point - targetPoint)
                result = np.sum(subtraction ** r)
                candidate.append((result ** index, idx))
            candidate = sorted(candidate, key=lambda x: x[0])
            res = [i[1] for i in candidate]

            return res[1:int(k + 1)]


def main():
    X = np.array([[1, 1], [8, 8], [9, 9], [10, 10], [7, 9], [13, 13], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
                  [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1],
                  [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])

    y = np.array([2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    smote_X, smote_y = Smote(X=X, Y=y, ratio=0.5, k=5, r=1).over_sampling()

    print('smote_X :', smote_X)
    print('smote_y :', smote_y)


if __name__ == '__main__':
    main()
