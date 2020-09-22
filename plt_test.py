import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

N = 1000
t = np.linspace(0,500000,num = N)
omega = math.pi/25
A = 10
x1 = A * np.sin(omega*t)
x2 = 0.5*A * np.sin(omega*(t-12500))

plt.rcParams["font.family"]="SimHei"  #添加此行可以全局显示中文，但无法正常显示坐标轴上的负号
plt.rcParams['axes.unicode_minus']=False  #再添加这一行就可以显示负号了

fig = plt.figure(num=1,figsize=(8,6))
gs = gridspec.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0,0:])  #ax1 就是这个画布上的子图，是一个类，可以调用plt的内部函数
ax1.set_xlabel("时间")
ax1.set_ylabel('函数值')
ax1.set_title('一个正弦函数')
ax1.plot(t,x1,'y--',label="正弦函数1")
ax1.plot(t,x2,'b--',label="正弦函数2")
ax1.legend(loc=1)

ax2 =fig.add_subplot(gs[1,0:1])
ax2.set_xlabel("时间")
ax2.set_ylabel("函数值")
ax2.plot(t,x1,'y--',label="正弦函数1")
ax2.legend(loc=1)

ax3 =fig.add_subplot(gs[1,1:])
ax3.set_xlabel("时间")
ax3.set_ylabel("函数值")
ax3.plot(t,x1,'b--',label="正弦函数2")
ax3.legend(loc=1)

plt.savefig("figure.jpg")
#自动保存方法，以后希望可以探索将具体的数据自动保存
plt.show()



