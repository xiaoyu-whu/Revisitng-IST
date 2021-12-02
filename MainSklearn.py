import importlib
import xlrd
import shutil
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import BayesianRidge
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from RegressionPerformanceMeasure import RegressionPerformanceMeasure
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from GPR import GPR
from SMOTE import Smote
from OverSampling import Oversampling
from IG import IG
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from Processing import Processing
from RegressionPerformanceMeasure import RegressionPerformanceMeasure
from configuration_file import configuration_file



header = ["dataset", "GP", "NNR", "DTR", "LR", "BRR", "SVR", "KNR", "GBR", "AR+DTR", "AR+LR","AR+BRR", "BG+DTR", "BG+LR", "BG+BRR","SR+DTR","SR+LR", "SR+BRR","OS+DTR","OS+LR", "OS+BRR", "IG+DTR","IG+LR", "IG+BRR", "GS+DTR","GS+LR", "GS+BRR"]

#the tuned parameters of DTR, LR, and BRR
dtr_tuned_parameters = [{'min_samples_split': [2, 3, 4, 5, 6]}]
lr_tuned_parameters = [{'normalize': [True, False]}]
brr_tuned_parameters = [{'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]}]
cv_times = 30

def DNP(training_data_X, training_data_y, testing_data_X, testing_data_y, dataset, trainingfilename,
              testingfilename):
    AAE0 = []
    AAE1 = []
    AAE2 = []
    AAE3 = []
    AAE456 = []
    AAEgt6 = []
    AAE = []
    predl0 = []
    predl1 = []
    predl2 = []
    predl3 = []
    predl456 = []
    predlgt6 = []
    predl = []


    AAE0.append(testingfilename)
    AAE1.append(testingfilename)
    AAE2.append(testingfilename)
    AAE3.append(testingfilename)
    AAE456.append(testingfilename)
    AAEgt6.append(testingfilename)
    AAE.append(testingfilename)
    predl0.append(testingfilename)
    predl1.append(testingfilename)
    predl2.append(testingfilename)
    predl3.append(testingfilename)
    predl456.append(testingfilename)
    predlgt6.append(testingfilename)
    predl.append(testingfilename)




    gpr = GPR(NP=200, F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)], generation=200, len_x=20,
                  value_up_range=2.0,
                  value_down_range=-2.0, X=training_data_X, y=training_data_y)
    gprmaxpara = gpr.process()
    gpr_pred_y = np.maximum(np.around(gpr.predict(testing_data_X, gprmaxpara)),0)
    gpr_pred_y = np.minimum(gpr_pred_y, 62)
    aae = RegressionPerformanceMeasure(testing_data_y, gpr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, gpr_pred_y).Predictionl(0.3)
    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    nnr = MLPRegressor(max_iter=1000).fit(training_data_X, training_data_y)
    nnr_pred_y = np.maximum(np.around(nnr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, nnr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, nnr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)
    dtr_pred_y = np.maximum(np.around(dtr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, dtr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # lr
    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
    lr_pred_y = np.maximum(np.around(lr.predict(testing_data_X)),0)
    aae = RegressionPerformanceMeasure(testing_data_y, lr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, lr_pred_y).Predictionl(0.3)
    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # brr
    brr = BayesianRidge().fit(training_data_X, training_data_y)
    brr_pred_y = np.maximum(np.around(brr.predict(testing_data_X)),0)
    aae = RegressionPerformanceMeasure(testing_data_y, brr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, brr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    svr = SVR().fit(training_data_X, training_data_y)
    svr_pred_y = np.maximum(np.around(svr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, svr_pred_y).AAE()
    pred=RegressionPerformanceMeasure(testing_data_y, svr_pred_y).Predictionl(0.3)
    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    knnr=KNeighborsRegressor().fit(training_data_X, training_data_y)
    knnr_pred_y = np.maximum(np.around(knnr.predict(testing_data_X)),0)
    aae = RegressionPerformanceMeasure(testing_data_y, knnr_pred_y).AAE()
    pred=RegressionPerformanceMeasure(testing_data_y, knnr_pred_y).Predictionl(0.3)
    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])
    
    
    # gbr
    gbr=GradientBoostingRegressor().fit(training_data_X, training_data_y)
    gbr_pred_y = np.maximum(np.around(gbr.predict(testing_data_X)),0)
    aae = RegressionPerformanceMeasure(testing_data_y, gbr_pred_y).AAE()
    pred=RegressionPerformanceMeasure(testing_data_y, gbr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])



    rng = np.random.RandomState(1)
    # boosting dtr
    dtrboosting = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=rng).fit(
        training_data_X, training_data_y)
    dtrboosting_pred_y = np.maximum(np.around(dtrboosting.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, dtrboosting_pred_y).AAE()
    pred=RegressionPerformanceMeasure(testing_data_y, dtrboosting_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])



    # boosting lr
    lrboosting = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, random_state=rng).fit(
        training_data_X, training_data_y)
    lrboosting_pred_y = np.maximum(np.around(lrboosting.predict(testing_data_X)),0)
    aae = RegressionPerformanceMeasure(testing_data_y, lrboosting_pred_y).AAE()
    pred=RegressionPerformanceMeasure(testing_data_y, lrboosting_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # boosting brr
    brrboosting = AdaBoostRegressor(BayesianRidge(), n_estimators=100, random_state=rng).fit(training_data_X,
                                                                                      training_data_y)
    brrboosting_pred_y = np.maximum(np.around(brrboosting.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, brrboosting_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, brrboosting_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # bagging dtr
    dtrbagging = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=rng).fit(
        training_data_X, training_data_y)
    dtrbagging_pred_y = np.maximum(np.around(dtrbagging.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, dtrbagging_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, dtrbagging_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # bagging lr
    lrbagging = BaggingRegressor(linear_model.LinearRegression(), n_estimators=100, random_state=rng).fit(
        training_data_X, training_data_y)
    lrbagging_pred_y = np.maximum(np.around(lrbagging.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, lrbagging_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, lrbagging_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    # bagging brr
    brrbagging = BaggingRegressor(BayesianRidge(), n_estimators=100, random_state=rng).fit(training_data_X,
                                                                                      training_data_y)
    brrbagging_pred_y = np.maximum(np.around(brrbagging.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, brrbagging_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, brrbagging_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    smote_X, smote_y = Smote(training_data_X, training_data_y, ratio=0.5, k=5, r=2).over_sampling()
    # smote dtr
    smotedtr = DecisionTreeRegressor().fit(smote_X, smote_y)
    smotedtr_pred_y = np.maximum(np.around(smotedtr.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, smotedtr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, smotedtr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    #smote lr
    smotelr = linear_model.LinearRegression().fit(smote_X, smote_y)
    smotelr_pred_y = np.maximum(np.around(smotelr.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, smotelr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, smotelr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    # smote brr
    smotebrr = BayesianRidge().fit(smote_X, smote_y)
    smotebrr_pred_y = np.maximum(np.around(smotebrr.predict(testing_data_X)),0)

    aae = RegressionPerformanceMeasure(testing_data_y, smotebrr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, smotebrr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    ros_X, ros_y = Oversampling(training_data_X, training_data_y, ratio=0.5).over_sampling()
    # ros dtr
    rosdtr = DecisionTreeRegressor().fit(ros_X, ros_y)
    rosdtr_pred_y = np.maximum(np.around(rosdtr.predict(testing_data_X)), 0)

    aae = RegressionPerformanceMeasure(testing_data_y, rosdtr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, rosdtr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    # ros lr
    roslr = linear_model.LinearRegression().fit(ros_X, ros_y)
    roslr_pred_y = np.maximum(np.around(roslr.predict(testing_data_X)), 0)

    aae = RegressionPerformanceMeasure(testing_data_y, roslr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, roslr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    # ros brr
    rosbrr = BayesianRidge().fit(ros_X, ros_y)
    rosbrr_pred_y = np.maximum(np.around(rosbrr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, rosbrr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, rosbrr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    #information gain

    IGtraining_data_X, IGtraining_data_y, IGtesting_data_X, IGtesting_data_y = IG(training_data_X, training_data_y, testing_data_X, testing_data_y).getSelectedFeature(4)

    IGdtr = DecisionTreeRegressor().fit(IGtraining_data_X, IGtraining_data_y)
    IGdtr_pred_y = np.maximum(np.around(IGdtr.predict(IGtesting_data_X)), 0)

    aae = RegressionPerformanceMeasure(IGtesting_data_y, IGdtr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(IGtesting_data_y, IGdtr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    # IG lr
    IGlr = linear_model.LinearRegression().fit(IGtraining_data_X, IGtraining_data_y)
    IGlr_pred_y = np.maximum(np.around(IGlr.predict(IGtesting_data_X)), 0)

    aae = RegressionPerformanceMeasure(IGtesting_data_y, IGlr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(IGtesting_data_y, IGlr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])

    # IG brr
    IGbrr = BayesianRidge().fit(IGtraining_data_X, IGtraining_data_y)
    IGbrr_pred_y = np.maximum(np.around(IGbrr.predict(IGtesting_data_X)), 0)
    aae = RegressionPerformanceMeasure(IGtesting_data_y, IGbrr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(IGtesting_data_y, IGbrr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    #grid search
    def my_score(realbug, predbug):
        defectaae=RegressionPerformanceMeasure(realbug, predbug).AAE()[1]+RegressionPerformanceMeasure(realbug, predbug).AAE()[2]+RegressionPerformanceMeasure(realbug, predbug).AAE()[3]+RegressionPerformanceMeasure(realbug, predbug).AAE()[4]+RegressionPerformanceMeasure(realbug, predbug).AAE()[5]
        return defectaae

    dtr = GridSearchCV(DecisionTreeRegressor(), dtr_tuned_parameters, cv=cv_times,
                       scoring=make_scorer(my_score, greater_is_better=False))
    dtr.fit(training_data_X, training_data_y)
    dtr_pred_y = np.maximum(np.around(dtr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, dtr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    lr = GridSearchCV(linear_model.LinearRegression(), lr_tuned_parameters, cv=cv_times,
                      scoring=make_scorer(my_score, greater_is_better=False))
    lr.fit(training_data_X, training_data_y)
    lr_pred_y = np.maximum(np.around(lr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, lr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, lr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])



    brr = GridSearchCV(BayesianRidge(), brr_tuned_parameters, cv=cv_times,
                         scoring=make_scorer(my_score, greater_is_better=False))
    brr.fit(training_data_X, training_data_y)
    brr_pred_y = np.maximum(np.around(brr.predict(testing_data_X)), 0)
    aae = RegressionPerformanceMeasure(testing_data_y, brr_pred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, brr_pred_y).Predictionl(0.3)

    AAE0.append(aae[0])
    AAE1.append(aae[1])
    AAE2.append(aae[2])
    AAE3.append(aae[3])
    AAE456.append(aae[4])
    AAEgt6.append(aae[5])
    AAE.append(aae[6])
    predl0.append(pred[0])
    predl1.append(pred[1])
    predl2.append(pred[2])
    predl3.append(pred[3])
    predl456.append(pred[4])
    predlgt6.append(pred[5])
    predl.append(pred[6])


    return AAE0, AAE1, AAE2, AAE3, AAE456, AAEgt6, AAE, predl0, predl1, predl2, predl3, predl456, predlgt6, predl



if __name__ == '__main__':


    AAE0_list = []
    AAE1_list = []
    AAE2_list = []
    AAE3_list = []
    AAE456_list = []
    AAEgt6_list = []
    AAE_list = []

    predl0_list = []
    predl1_list = []
    predl2_list = []
    predl3_list = []
    predl456_list = []
    predlgt6_list = []
    predl_list = []


    AAE0_list.append(header)
    AAE1_list.append(header)
    AAE2_list.append(header)
    AAE3_list.append(header)
    AAE456_list.append(header)
    AAEgt6_list.append(header)
    AAE_list.append(header)

    predl0_list.append(header)
    predl1_list.append(header)
    predl2_list.append(header)
    predl3_list.append(header)
    predl456_list.append(header)
    predlgt6_list.append(header)
    predl_list.append(header)




    print("========================")
    for training_data_X, training_data_y, testing_data_X, testing_data_y, dataset, trainingfilename, testingfilename in Processing().import_crossversion_data():
        print("---------------------")
        print('trainingdata', trainingfilename, '   testingdata', testingfilename)

        AAE0, AAE1, AAE2, AAE3, AAE456, AAEgt6, AAE, predl0, predl1, predl2, predl3, predl456, predlgt6, predl = DNP(
            training_data_X, training_data_y, testing_data_X, testing_data_y, dataset, trainingfilename, testingfilename)

        AAE0_list.append(AAE0)
        AAE1_list.append(AAE1)
        AAE2_list.append(AAE2)
        AAE3_list.append(AAE3)
        AAE456_list.append(AAE456)
        AAEgt6_list.append(AAEgt6)
        AAE_list.append(AAE)

        predl0_list.append(predl0)
        predl1_list.append(predl1)
        predl2_list.append(predl2)
        predl3_list.append(predl3)
        predl456_list.append(predl456)
        predlgt6_list.append(predlgt6)
        predl_list.append(predl)


    result_path = configuration_file().performancemeasureresult

    aae0_csv_name = "sklearnpackageAAE0.xlsx"
    aae0result_path = os.path.join(result_path, aae0_csv_name)
    Processing().write_excel(aae0result_path, AAE0_list)

    aae1_csv_name = "sklearnpackageAAE1.xlsx"
    aae1result_path = os.path.join(result_path, aae1_csv_name)
    Processing().write_excel(aae1result_path, AAE1_list)

    aae2_csv_name = "sklearnpackageAAE2.xlsx"
    aae2result_path = os.path.join(result_path, aae2_csv_name)
    Processing().write_excel(aae2result_path, AAE2_list)

    aae3_csv_name = "sklearnpackageAAE3.xlsx"
    aae3result_path = os.path.join(result_path, aae3_csv_name)
    Processing().write_excel(aae3result_path, AAE3_list)

    aae456_csv_name = "sklearnpackageAAE456.xlsx"
    aae456result_path = os.path.join(result_path, aae456_csv_name)
    Processing().write_excel(aae456result_path, AAE456_list)

    aaegt6_csv_name = "sklearnpackageAAEgt6.xlsx"
    aaegt6result_path = os.path.join(result_path, aaegt6_csv_name)
    Processing().write_excel(aaegt6result_path, AAEgt6_list)

    aae_csv_name = "sklearnpackageAAE.xlsx"
    aaeresult_path = os.path.join(result_path, aae_csv_name)
    Processing().write_excel(aaeresult_path, AAE_list)

    predl0_csv_name = "sklearnpackagepredl0.xlsx"
    predl0result_path = os.path.join(result_path, predl0_csv_name)
    Processing().write_excel(predl0result_path, predl0_list)

    predl1_csv_name = "sklearnpackagepredl1.xlsx"
    predl1result_path = os.path.join(result_path, predl1_csv_name)
    Processing().write_excel(predl1result_path, predl1_list)

    predl2_csv_name = "sklearnpackagepredl2.xlsx"
    predl2result_path = os.path.join(result_path, predl2_csv_name)
    Processing().write_excel(predl2result_path, predl2_list)

    predl3_csv_name = "sklearnpackagepredl3.xlsx"
    predl3result_path = os.path.join(result_path, predl3_csv_name)
    Processing().write_excel(predl3result_path, predl3_list)

    predl456_csv_name = "sklearnpackagepredl456.xlsx"
    predl456result_path = os.path.join(result_path, predl456_csv_name)
    Processing().write_excel(predl456result_path, predl456_list)

    predlgt6_csv_name = "sklearnpackagepredlgt6.xlsx"
    predlgt6result_path = os.path.join(result_path, predlgt6_csv_name)
    Processing().write_excel(predlgt6result_path, predlgt6_list)

    predl_csv_name = "sklearnpackagepredl.xlsx"
    predlresult_path = os.path.join(result_path, predl_csv_name)
    Processing().write_excel(predlresult_path, predl_list)