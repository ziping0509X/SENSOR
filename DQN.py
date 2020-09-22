import numpy as np
import  tensorflow as tf
import  time
import  random
from collections import  deque


GAMMA = 0.8 #损失系数
OBSERVE = 300
EXPLORE = 60000
FINAL_EPSILON = 0.0
INITIAL_EPSILON = 0.8
REPLAY_MEMORY = 400
BATCH_SIZE = 256
SENSOR_NUM = 10
ACTION_NUM = 26

class BrainDQN:

    def __init__(self,SENSOR_NUM,ACTION_NUM):

        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.recording = EXPLORE
        self.sensor_num = SENSOR_NUM
        self.action_num = ACTION_NUM
        self.hidden1 = 32
        self.hidden2 = 64
        self.hidden3 = 32

        self.createQNetwork()

    def weight_variable(self, shape):
         initial = tf.truncated_normal(shape)
         return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def createQNetwork(self):

        #需要feed的数据是state和action
        self.stateInput = tf.placeholder("float", [None, self.sensor_num])
        #self.actionInput = tf.placeholder("float", [None, self.action_num])

        W_fc1 = self.weight_variable([self.sensor_num, self.hidden1])
        b_fc1 = self.bias_variable([self.hidden1])

        W_fc2 = self.weight_variable([self.hidden1, self.hidden2])
        b_fc2 = self.bias_variable([self.hidden2])

        W_fc3 = self.weight_variable([self.hidden2, self.hidden3])
        b_fc3 = self.bias_variable([self.hidden3])

        W_fc4 = self.weight_variable([self.hidden3, self.action_num])
        b_fc4 = self.bias_variable([self.action_num])

        h_1 = tf.nn.relu(tf.matmul(self.stateInput,W_fc1)+b_fc1)
        h_2 = tf.nn.relu(tf.matmul(h_1, W_fc2) + b_fc2)
        h_3 = tf.nn.relu(tf.matmul(h_2, W_fc3) + b_fc3)

        self.Qvalue = tf.matmul(h_3,W_fc4)+b_fc4

        self.actionInput = tf.placeholder("float", [None, self.action_num])
        self.Qvalue_T = tf.placeholder("float",[None])
        #Qvalue_target

        Q_action = tf.multiply(self.Qvalue,self.actionInput)
        Q_action = tf.reduce_sum(Q_action,reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.Qvalue_T-Q_action))
        #输入的是batch_size=128的数组，然后求出128个差距值，最后求出一个平均值来
        self.trainStep = tf.train.AdamOptimizer(learning_rate = 10**-6).minimize(self.cost)
        #初始化语句
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def trainQNetwork(self):
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        state_batch = np.array(state_batch)
        state_batch = state_batch.reshape(256, 10)
        action_batch = [data[2] for data in minibatch]
        reward_batch = [data[3] for data in minibatch]
        nextState_batch = [data[1] for data in minibatch]
        nextState_batch = np.array(nextState_batch)
        nextState_batch = nextState_batch.reshape(256,10)



        Qvalue_T_batch = []
        Qvalue_batch = self.Qvalue.eval(feed_dict={self.stateInput:nextState_batch})

        for i in range(0,BATCH_SIZE):
            Qvalue_T_batch.append(reward_batch[i]+GAMMA * np.max(Qvalue_batch))
        _, self.loss = self.session.run([self.trainStep,self.cost],feed_dict={
                        self.actionInput : action_batch,
                        self.stateInput : state_batch,
                        self.Qvalue_T : Qvalue_T_batch})
        return self.loss


    def getAction(self,action_num,stateInput):#使用贪心策略决定动作选择
        action = np.zeros(action_num)
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action_num)
            action[action_index] = 1
            #print("use random strategy:")
            print(action_index)
        else:
            Qvalue = self.Qvalue.eval(feed_dict={self.stateInput:stateInput}) #有了self前缀才可以在class中无差别地调用
            #print("use max Q-value:")
            #print(np.max(Qvalue))
            action_index = np.argmax(Qvalue)
            print([action_index])
            action[action_index] = 1
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def getAction_1(self,action_num,stateInput):#使用贪心策略决定动作选择
        action = np.zeros(action_num)

        Qvalue = self.Qvalue.eval(feed_dict={self.stateInput:stateInput}) #有了self前缀才可以在class中无差别地调用
        print("the max Q-value is:")
        print(np.max(Qvalue))
        action_index = np.argmax(Qvalue)
        action[action_index] = 1
        print("use max Q-value:")

        return action

    def get_loss(self,currentState,nextState,action,reward):
        loss = 0
        self.currentState = currentState
        newState = nextState
        self.replayMemory.append((currentState,nextState,action,reward))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            loss = self.trainQNetwork()


        self.timeStep += 1

        self.currentState = newState

        return loss
























