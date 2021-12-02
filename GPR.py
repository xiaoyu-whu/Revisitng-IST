# # -*- coding: utf-8 -*-
# @Author: xiao yu
# @Date:   2021-5-20
# @Last Modified by:   chenkai shen
# @Last Modified time: 2021-7-10
#genetic programing
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split
from Processing import Processing


class GPR:
    def __init__(self,
                 NP=200,
                 F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)],
                 generation=200,
                 len_x=20,
                 value_up_range=20.0,
                 value_down_range=-20.0,
                 X=None,
                 y=None):

        self.NP = NP
        self.F_CR = F_CR
        self.generation = generation
        self.len_x = 20
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range

        self.np_list = self.initialtion()
        self.training_data_X = X
        self.training_data_y = y

    def initialtion(self):

        np_list = []
        for i in range(0, self.NP):
            x_list = []
            for j in range(0, self.len_x):
                x_list.append(self.value_down_range + random.random() *
                              (self.value_up_range - self.value_down_range))
            np_list.append(x_list)

        return np_list

    def substract(self, a_list, b_list):

        return [a - b for (a, b) in zip(a_list, b_list)]

    def add(self, a_list, b_list):

        return [a + b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):

        return [a * b for b in b_list]

    def random_distinct_integers(self, number):

        res = set()
        while len(res) != int(number):
            res.add(random.randint(0, self.NP - 1))
        return list(res)

    def mutation_crossover_one(self, np_list):

        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]

            sub = self.substract(np_list[r2], np_list[r3])
            mul = self.multiply(F, sub)
            add = self.add(np_list[r1], mul)

            for i in range(self.len_x):
                if add[i] > self.value_up_range or add[i] < self.value_down_range:
                    add[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)

            v_list.append(add)

        # crossover
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):
                if (random.random() <= CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])

            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def mutation_crossover_two(self, np_list):

        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]
        F1 = random.random()

        v_list = []
        for i in range(0, self.NP):
            r12345 = self.random_distinct_integers(5)
            r1 = r12345[0]
            r2 = r12345[1]
            r3 = r12345[2]
            r4 = r12345[3]
            r5 = r12345[4]

            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r4], np_list[r5])
            mul1 = self.multiply(F1, sub1)
            mul2 = self.multiply(F, sub2)
            add1 = self.add(np_list[r1], mul1)
            add2 = self.add(add1, mul2)

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_three(self, np_list):

        F_CR = random.choice(self.F_CR)
        F = F_CR[0]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]
            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r1], np_list[i])
            mul1 = self.multiply(F, sub1)
            mul2 = self.multiply(random.random(), sub2)
            add1 = self.add(mul1, mul2)
            add2 = self.add(add1, np_list[i])

            for i in range(self.len_x):
                if add2[i] > self.value_up_range or add2[i] < self.value_down_range:
                    add2[i] = self.value_down_range + random.random() * (
                            self.value_up_range - self.value_down_range)
            v_list.append(add2)

        return v_list

    def crossover(self, np_list, v_list, CR):

        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):
                if (random.random() <= CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])

            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list1, u_list2, u_list3, np_list):

        for i in range(0, self.NP):
            fpa1 = self.Objfunction(u_list1[i])
            fpa2 = self.Objfunction(u_list2[i])
            fpa3 = self.Objfunction(u_list3[i])
            fpa4 = self.Objfunction(np_list[i])
            max_fpa = max(fpa1, fpa2, fpa3, fpa4)
            if max_fpa == fpa1:
                np_list[i] = u_list1[i]
            elif max_fpa == fpa2:
                np_list[i] = u_list2[i]
            elif max_fpa == fpa3:
                np_list[i] = u_list3[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def process(self):
        np_list = self.np_list
        max_x = []
        max_f = []
        for i in range(0, self.NP):
            xx = []
            xx.append(self.Objfunction(np_list[i]))
        max_f.append(max(xx))
        max_x.append(np_list[xx.index(max(xx))])
        for i in range(0, self.generation):
            #print("iteration {0}".format(i))
            u_list1 = self.mutation_crossover_one(np_list)
            u_list2 = self.mutation_crossover_two(np_list)
            u_list3 = self.mutation_crossover_three(np_list)
            np_list = self.selection(u_list1, u_list2, u_list3, np_list)
            for i in range(0, self.NP):
                xx = []
                xx.append(self.Objfunction(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        max_ff = max(max_f)
        max_xx = max_x[max_f.index(max_ff)]
       # print('the maximum point x =', max_xx)
       # print('the maximum value y =', max_ff)

        '''
        x_label = np.arange(0, self.generation + 1, 1)
        plt.plot(x_label, max_f, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('./iteration-f.png')
        plt.show()
        '''

        return max_xx


    def Objfunction(self, Param):
        pred_y = []
        for test_x in self.training_data_X:

            pred_y.append(float(np.dot(test_x, Param)))

        are = 0
        for i in range(len(self.training_data_y)):
            are = are + (abs(self.training_data_y[i] - pred_y[i]) * -1.0) / (self.training_data_y[i] + 1)
        are = are / len(self.training_data_y)

        return are



    def predict(self, testing_data_X, Param):
        pred_y = []
        for test_x in testing_data_X:
            pred_y.append(float(np.dot(test_x, Param)))

        return pred_y

if __name__ == '__main__':

    for dataset, filename in Processing().import_single_data():
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)
        gpr = GPR(NP=200, F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)], generation=1000, len_x=20,
                  value_up_range=20.0,
                  value_down_range=-20.0, X=training_data_X, y=training_data_y)
        gpr.process()