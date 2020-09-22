import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt


# class ENVIRONMNET:
#     def __init__(self):

N = 100000
t = np.linspace(0,100000,num = N)  #相当于matlab中的向量，本质上横轴是一个离散的点
omega = math.pi/25
A = 10
x1 = A * np.sin(omega*t)

action_size = 26

def action_put(x1):
    action_put = []
    for i in range(13):
        action_put.append(x1[i])
    for i in range(25,38):
        action_put.append(x1[i])
    return action_put
K = action_put(x1)
print(K)

class ENVIRNOMENT:
    def __init__(self,action_size):
        self.action_size = action_size

    def creat_sensor(self,Power,sensor_num):
        stateInput = np.zeros(sensor_num)
        for i in range(sensor_num):
            stateInput[i] = Power
        stateInput = stateInput.reshape(1,-1)
        return stateInput

    def get_reward(self,stateInput,actionInput):
        Power = stateInput[0][0]
        Action = list(actionInput).index(1)
        Action = K[Action]
        temp = abs(Power + Action) // 1  #temp是一个[1,1]的数组，不是一个整形的数据

        reward = 0
        if  not temp:
            reward = 1
        if temp:
            reward = 0
        return reward












