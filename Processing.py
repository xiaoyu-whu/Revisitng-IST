
import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
from configuration_file import configuration_file
from openpyxl import Workbook


class Processing():
    def __init__(self):

        self.crossversionfolder_name = configuration_file().crossversiondatafolderPath


    def import_crossversion_data(self):

        dataset_train = pd.core.frame.DataFrame()
        dataset_test = pd.core.frame.DataFrame()

        folder_path = self.crossversionfolder_name + '/'

        def transform_data(original_data):
           # original_data = original_data.iloc[:, 3:]

            original_data = np.array(original_data)

            k = len(original_data[0])

            original_data = sorted(
                original_data, key=lambda x: x[-1], reverse=True)

            original_data = np.array(original_data)
            original_data_X = original_data[:, 0:k - 1]

            original_data_y = original_data[:, k - 1]

            return original_data_X, original_data_y

        for root, dirs, files, in os.walk(folder_path):
            if root == folder_path:

                thisroot = root
                for dir in dirs:
                    dir_path = os.path.join(thisroot, dir)

                    for root, dirs, files, in os.walk(dir_path):
                        if(files[0][-7:-4]<files[1][-7:-4]):
                            file_path_train = os.path.join(dir_path, files[0])
                            file_path_test = os.path.join(dir_path, files[1])
                            trainingfile=files[0]
                            testingfile=files[1]
                        else:
                            file_path_train = os.path.join(dir_path, files[1])
                            file_path_test = os.path.join(dir_path, files[0])
                            trainingfile = files[1]
                            testingfile = files[0]

                        dataset_train = pd.read_csv(file_path_train)
                        dataset_test = pd.read_csv(file_path_test)
                        training_data_x, training_data_y = transform_data(
                            dataset_train)
                        testing_data_x, testing_data_y = transform_data(
                            dataset_test)
                        yield training_data_x, training_data_y, testing_data_x, testing_data_y, dir, trainingfile, testingfile


    def write_excel(self, excel_path, data):
        # try:
        dir_name = str(os.path.split(excel_path)[0])
        print(dir_name)
        #mkdir(dir_name)
        wb = Workbook()
        ws = wb.active
        for _ in data:
            ws.append(_)
        wb.save(excel_path)
