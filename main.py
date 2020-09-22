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
time = 0

#构造状态生成函数A*sin(omega*t)
N = 100000
t = np.linspace(0,100000,num = N) #步进为1，共采样10000个数据
omega = math.pi/25
A = 10
x1 = A * np.sin(omega*t)

DQN = BrainDQN(SENSOR_NUM,ACTION_NUM)
ENV = ENVIRNOMENT(ACTION_NUM)
R = []
R_total = 0
while not record ==0:
    if time >90000:
        break
    state = x1[time]
    stateInput = ENV.creat_sensor(Power= x1[time],sensor_num= SENSOR_NUM)
    action_input = DQN.getAction(action_num= ACTION_NUM,stateInput= stateInput)
    reward = ENV.get_reward(stateInput= stateInput,actionInput= action_input)
    #print(reward)
    R_total += reward
    R.append(R_total)
    nextState = ENV.creat_sensor(Power= x1[time+1],sensor_num= SENSOR_NUM)

    #将stateinput、actioninput、reward、nextState放入replaymemory中
    #LOSS.append(DQN.get_loss(currentState= stateInput,nextState= nextState,action=action_input,reward= reward))
    loss = DQN.get_loss(currentState=stateInput, nextState=nextState, action=action_input, reward=reward)
    LOSS.append(loss)
    time = time + 1
    record = record -1
    #print(record)

plt.rcParams["font.family"]="SimHei"  #添加此行可以全局显示中文，但无法正常显示坐标轴上的负号
# plt.rcParams['axes.unicode_minus']=False  #再添加这一行就可以显示负号了
#
# #显示输入状态集
# fig = plt.figure(num=1,figsize=(8,6))
# ax1 = fig.add_subplot(111)
# ax1.set_xlim(0,200)
# ax1.set_xlabel("时间")
# ax1.set_ylabel('函数值')
# ax1.set_title('输入状态集')
# ax1.plot(t,x1,'b--',label="输入状态集")
# ax1.legend(loc=1)
plt.plot(LOSS)
plt.figure()
plt.plot(R)
plt.show()





