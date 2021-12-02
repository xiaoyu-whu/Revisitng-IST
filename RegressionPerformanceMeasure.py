# -*- coding: utf-8 -*-
# @Author: xiao yu
# @Date:   2021-5-20
# @Last Modified by:   chenkai shen
# @Last Modified time: 2021-11-7
# Evaluate the AAE and Pred(0.3) values for the modules with different number of defects separately.
import numpy as np
from sklearn import metrics

class RegressionPerformanceMeasure():
    def __init__(self, real_list, pred_list=None):
        self.real = real_list
        self.pred = pred_list

    def AAE(self):

        AAE0 = 0;
        count0 = 0;
        AAE1 = 0;
        count1 = 0;
        AAE2 = 0;
        count2 = 0;
        AAE3 = 0;
        count3 = 0;
        AAE456 = 0;
        count456 = 0;
        AAEgt6 = 0;
        countgt6 = 0;

        for i in range(len(self.real)):
            if self.real[i] == 0:
                AAE0 = AAE0 + (abs(self.real[i] - self.pred[i]))
                count0 = count0 + 1
            elif self.real[i] == 1:
                AAE1 = AAE1 + (abs(self.real[i] - self.pred[i]))
                count1 = count1 + 1
            elif self.real[i] == 2:
                AAE2 = AAE2 + (abs(self.real[i] - self.pred[i]))
                count2 = count2 + 1
            elif self.real[i] == 3:
                AAE3 = AAE3 + (abs(self.real[i] - self.pred[i]))
                count3 = count3 + 1
            elif 3 < self.real[i] < 7:
                AAE456 = AAE456 + (abs(self.real[i] - self.pred[i]))
                count456 = count456 + 1
            elif self.real[i] > 6:
                AAEgt6 = AAEgt6 + (abs(self.real[i] - self.pred[i]))
                countgt6 = countgt6 + 1

        AAE = AAE0 + AAE1 + AAE2 + AAE3 + AAE456 + AAEgt6
        AAE = AAE / (len(self.real) * 1.0)

        if count0!=0:
            AAE0 = AAE0 / (count0 * 1.0)
        else:
            AAE0 = 999

        if count1 != 0:
            AAE1 = AAE1 / (count1 * 1.0)
        else:
            AAE1 = 999
        if count2 != 0:
            AAE2 = AAE2 / (count2 * 1.0)
        else:
            AAE2 = 999
        if count3 != 0:
            AAE3 = AAE3 / (count3 * 1.0)
        else:
            AAE3 = 999
        if count456 != 0:
            AAE456 = AAE456 / (count456 * 1.0)
        else:
            AAE456 = 999
        if countgt6 != 0:
            AAEgt6 = AAEgt6 / (countgt6 * 1.0)
        else:
            AAEgt6 = 999

        return AAE0, AAE1, AAE2, AAE3, AAE456, AAEgt6, AAE

    def Predictionl(self, l):

        predl0 = 0;
        count0 = 0;
        predl1 = 0;
        count1 = 0;
        predl2 = 0;
        count2 = 0;
        predl3 = 0;
        count3 = 0;
        predl456 = 0;
        count456 = 0;
        predlgt6 = 0;
        countgt6 = 0;

        for i in range(len(self.real)):
            if self.real[i] == 0:
                count0 = count0 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predl0 = predl0 + 1
            elif self.real[i] == 1:
                count1 = count1 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predl1 = predl1 + 1
            elif self.real[i] == 2:
                count2 = count2 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predl2 = predl2 + 1
            elif self.real[i] == 3:
                count3 = count3 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predl3 = predl3 + 1
            elif 3 < self.real[i] < 7:
                count456 = count456 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predl456 = predl456 + 1
            elif self.real[i] > 6:
                countgt6 = countgt6 + 1
                if abs(self.real[i] - self.pred[i]) < l * (1 + self.real[i]):
                    predlgt6 = predlgt6 + 1

        predl = predl0 + predl1 + predl2 + predl3 + predl456 + predlgt6
        predl = predl / (len(self.real) * 1.0)

        predl0 = predl0 / (count0 * 1.0)
        if count1 != 0:
            predl1 = predl1 / (count1 * 1.0)
        else:
            predl1 = 999
        if count2 != 0:
            predl2 = predl2 / (count2 * 1.0)
        else:
            predl2 = 999
        if count3 != 0:
            predl3 = predl3 / (count3 * 1.0)
        else:
            predl3 = 999
        if count456 != 0:
            predl456 = predl456 / (count456 * 1.0)
        else:
            predl456 = 999
        if countgt6 != 0:
            predlgt6 = predlgt6 / (countgt6 * 1.0)
        else:
            predlgt6 = 999

        return predl0, predl1, predl2, predl3, predl456, predlgt6, predl



if __name__ == '__main__':
    real = np.array([5, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred1 = np.array([5, 3, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    real_list = np.array([2, 3, 0, 0, 1, 1, 0, 5, 3])
    pred_list = np.array([1, 1, 1, 0, 1, 0, 0, 3, 4])

    aae_result = RegressionPerformanceMeasure(real_list, pred_list).AAE()
    predl_result = RegressionPerformanceMeasure(real_list, pred_list).Predictionl(l=0.3)
    print(aae_result)
    print(predl_result)
