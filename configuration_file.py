import os


class configuration_file():
    def __init__(self):
        # the root path of the whole project
        self.rootpath = r"/Users/xiao/Documents/project/Python/Revisiting-IST"

        #the path of predicted number of defect of 5 regression algorithm using R package
        self.Rresult = r"/Users/xiao/Documents/project/Python/Revisiting-IST/Rresult"

        #the path of the promise dataset
        self.crossversiondatafolderPath = os.path.join(self.rootpath, "CrossversionData1")

        #the result path of all performance measures
        self.performancemeasureresult = os.path.join(self.rootpath, "test_result")

        pass

    def getrootpath(self, a):
        return self.rootpath
