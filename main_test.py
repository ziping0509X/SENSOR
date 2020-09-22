import DQN
from DQN import BrainDQN
import random
import time
import numpy as np
import ENVIRONMENT
from ENVIRONMENT import ENVIRNOMENT
import math
from matplotlib import pyplot as plt
import matplotlib

LOSS = []

SENSOR_NUM = 10
ACTION_NUM = 26
record = 95000
Time = 10

#构造状态生成函数A*sin(omega*t)
N = 10000
t = np.linspace(0,10000,num = N) #步进为1，共采样10000个数据
omega = math.pi/25
A = 10
x1 = A * np.sin(omega*t)

DQN1 = BrainDQN(SENSOR_NUM,ACTION_NUM)
ENV = ENVIRNOMENT(ACTION_NUM)

state = x1[Time]
stateInput = ENV.creat_sensor(Power= x1[Time],sensor_num= SENSOR_NUM) #通过

action_input = DQN1.getAction_1(action_num= ACTION_NUM,stateInput= stateInput)
print(action_input)

# reward = ENV.get_reward(stateInput= stateInput,actionInput= action_input)
# nextState = ENV.creat_sensor(Power= x1[time+1],sensor_num= SENSOR_NUM)
#
# #将stateinput、actioninput、reward、nextState放入replaymemory中
# LOSS.append(DQN1.get_loss(currentState= stateInput,nextState= nextState,action=action_input,reward= reward))
# time = time + 1
# record = record -1