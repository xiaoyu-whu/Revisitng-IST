import numpy as np
import math

class IG():
    def __init__(self,X,y,testX,testy):

        X = np.array(X)
        y = list(y)
        n_feature = np.shape(X)[1]
        n_y = len(y)

        orig_H = 0
        for i in set(y):
            orig_H += -(y.count(i)/n_y)*math.log(y.count(i)/n_y)


        condi_H_list = []
        for i in range(n_feature):
            feature = X[:,i]
            sourted_feature = sorted(feature)
            threshold = [(sourted_feature[inde-1]+sourted_feature[inde])/2 for inde in range(len(feature)) if inde != 0 ]

            thre_set = set(threshold)
            if float(max(feature)) in thre_set:
                thre_set.remove(float(max(feature)))
            if float(min(feature)) in thre_set:
                thre_set.remove(float(min(feature)))
            pre_H = 0
            maxTHre = 0
            for thre in thre_set:
                lower = [y[s] for s in range(len(feature)) if feature[s] < thre]
                highter = [y[s] for s in range(len(feature)) if feature[s] > thre]
                H_l = 0
                for l in set(lower):
                    prob = lower.count(l) / len(lower)
                    H_l += -prob*math.log(prob)
                H_h = 0
                for h in set(highter):
                    prob = highter.count(h) / len(highter)
                    H_h += -prob*math.log(prob)
                temp_condi_H = len(lower)/n_y *H_l+ len(highter)/n_y * H_h
                condi_H = orig_H - temp_condi_H
                pre_H = max(pre_H,condi_H)
                maxTHre = max(maxTHre,thre)
            condi_H_list.append(pre_H)

        self.IG = condi_H_list
        self.Y =  y
        index = list(np.argsort(condi_H_list))
        index.reverse()
        testX = np.array(testX)
        self.testy = np.array(testy)
        self.feature_X = X[:,index]
        self.test_X = testX[:,index]


    def getIG(self):
        return self.IG

    def getSelectedFeature(self,top_k):

        return self.feature_X[:,:top_k],self.Y,self.test_X[:,:top_k],self.testy




if __name__ == "__main__":


    X = [[1, 0, 0, 1],
         [0, 1, 1, 1],
         [0, 0, 1, 0]]
    y = [0, 0, 1]

    test_X = [[1, 0, 0, 1],
         [0, 1, 1, 1],
         [0, 0, 1, 0]]
    test_y = [0, 0, 1]

    #X = [[0.1],[0.5],[0.3],[0.7],[0.8],[0.8]]
    #y = [1,1,2,2,3,3]

    #print(IG(X,y).getIG())

    #print((-1/3*math.log(1/3)*3)-(2/3*(-math.log(1/2))))

    X , y,testX,testy =IG(X,y,test_X,test_y).getSelectedFeature(2)
    print(X,y)
    print(testX,testy)