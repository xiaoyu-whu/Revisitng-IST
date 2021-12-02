import importlib
from Processing import Processing
from RegressionPerformanceMeasure import RegressionPerformanceMeasure
from configuration_file import configuration_file
import numpy as np
import os
import pandas as pd

header = ["dataset", "PR", "ZIPR", "NBR", "ZINBR", "HR"]


def RDNP(testing_data_y, HPRpred_path, NBRpred_path, PRpred_path, ZINBRpred_path, ZIPRpred_path, testingfilename):
    HPRpred_y = read_predY(HPRpred_path)
    NBRpred_y = read_predY(NBRpred_path)
    PRpred_y = read_predY(PRpred_path)
    ZINBRpred_y = read_predY(ZINBRpred_path)
    ZIPRpred_y = read_predY(ZIPRpred_path)

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


    aae = RegressionPerformanceMeasure(testing_data_y, PRpred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, PRpred_y).Predictionl(0.3)
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


    aae = RegressionPerformanceMeasure(testing_data_y, ZIPRpred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, ZIPRpred_y).Predictionl(0.3)
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

    aae = RegressionPerformanceMeasure(testing_data_y, NBRpred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, NBRpred_y).Predictionl(0.3)
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

    aae = RegressionPerformanceMeasure(testing_data_y, ZINBRpred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, ZINBRpred_y).Predictionl(0.3)
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


    aae = RegressionPerformanceMeasure(testing_data_y, HPRpred_y).AAE()
    pred = RegressionPerformanceMeasure(testing_data_y, HPRpred_y).Predictionl(0.3)
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


def read_predY(pred_path):
    dataset_train = pd.read_csv(pred_path)
    pred_y = dataset_train.iloc[:, 1]
    #the predicted number of defects should be an integer and not less than zero
    pred_y = np.maximum(np.around(pred_y), 0)
    #the maxmnium number of defects in the dataset is 62, so we need the predicted number is not larger than 62.
    pred_y = np.minimum(pred_y, 62)
    return np.array(pred_y)

    pass


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

        Rresult_folder = configuration_file().Rresult

        PRpred_path = os.path.join(Rresult_folder, 'PR' + testingfilename)
        ZIPRpred_path = os.path.join(Rresult_folder, 'ZIPR' + testingfilename)
        NBRpred_path = os.path.join(Rresult_folder, 'NBR' + testingfilename)
        ZINBRpred_path = os.path.join(Rresult_folder, 'ZINBR' + testingfilename)
        HPRpred_path = os.path.join(Rresult_folder, 'HR' + testingfilename)


        AAE0, AAE1, AAE2, AAE3, AAE456, AAEgt6, AAE, predl0, predl1, predl2, predl3, predl456, predlgt6, predl = RDNP(
            testing_data_y, HPRpred_path, NBRpred_path, PRpred_path, ZINBRpred_path, ZIPRpred_path, testingfilename)

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

    aae0_csv_name = "RpackageAAE0.xlsx"
    aae0result_path = os.path.join(result_path, aae0_csv_name)
    Processing().write_excel(aae0result_path, AAE0_list)

    aae1_csv_name = "RpackageAAE1.xlsx"
    aae1result_path = os.path.join(result_path, aae1_csv_name)
    Processing().write_excel(aae1result_path, AAE1_list)

    aae2_csv_name = "RpackageAAE2.xlsx"
    aae2result_path = os.path.join(result_path, aae2_csv_name)
    Processing().write_excel(aae2result_path, AAE2_list)

    aae3_csv_name = "RpackageAAE3.xlsx"
    aae3result_path = os.path.join(result_path, aae3_csv_name)
    Processing().write_excel(aae3result_path, AAE3_list)

    aae456_csv_name = "RpackageAAE456.xlsx"
    aae456result_path = os.path.join(result_path, aae456_csv_name)
    Processing().write_excel(aae456result_path, AAE456_list)

    aaegt6_csv_name = "RpackageAAEgt6.xlsx"
    aaegt6result_path = os.path.join(result_path, aaegt6_csv_name)
    Processing().write_excel(aaegt6result_path, AAEgt6_list)

    aae_csv_name = "RpackageAAE.xlsx"
    aaeresult_path = os.path.join(result_path, aae_csv_name)
    Processing().write_excel(aaeresult_path, AAE_list)

    predl0_csv_name = "Rpackagepredl0.xlsx"
    predl0result_path = os.path.join(result_path, predl0_csv_name)
    Processing().write_excel(predl0result_path, predl0_list)

    predl1_csv_name = "Rpackagepredl1.xlsx"
    predl1result_path = os.path.join(result_path, predl1_csv_name)
    Processing().write_excel(predl1result_path, predl1_list)

    predl2_csv_name = "Rpackagepredl2.xlsx"
    predl2result_path = os.path.join(result_path, predl2_csv_name)
    Processing().write_excel(predl2result_path, predl2_list)

    predl3_csv_name = "Rpackagepredl3.xlsx"
    predl3result_path = os.path.join(result_path, predl3_csv_name)
    Processing().write_excel(predl3result_path, predl3_list)

    predl456_csv_name = "Rpackagepredl456.xlsx"
    predl456result_path = os.path.join(result_path, predl456_csv_name)
    Processing().write_excel(predl456result_path, predl456_list)

    predlgt6_csv_name = "Rpackagepredlgt6.xlsx"
    predlgt6result_path = os.path.join(result_path, predlgt6_csv_name)
    Processing().write_excel(predlgt6result_path, predlgt6_list)

    predl_csv_name = "Rpackagepredl.xlsx"
    predlresult_path = os.path.join(result_path, predl_csv_name)
    Processing().write_excel(predlresult_path, predl_list)