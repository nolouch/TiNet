'''
预测数据,多维单步，预测几步后的那一时刻输出
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import *
from math import sqrt
import time
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#获取训练集
def get_train_data(data,batch_size,time_step,train_end,train_begin=0):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    maxvalue = np.max(data_train, axis=0)
    normalized_train_data = data_train / np.max(data_train, axis=0)

    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step-(predict_step-1)):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step]
       y = normalized_train_data[i+time_step+(predict_step-1)]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step-(predict_step-1)))
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(data,time_step,test_begin):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    #normalized_test_data=(data_test-mean)/std  #标准化
    maxvalue = np.max(data_test, axis=0)
    #print(maxvalue)
    #maxvalue += 1
    #print(maxvalue)
    normalized_test_data = data_test / maxvalue
    #size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]
    for i in range(len(normalized_test_data)-time_step-(predict_step-1)):
       x=normalized_test_data[i:i+time_step]
       y=normalized_test_data[i+time_step+(predict_step-1)]
       test_x.append(x.tolist())
       test_y.append(y)
    print('get_test_data x',np.array(test_x).shape)
    #print(np.array(test_x))
    print('get_test_data y', np.array(test_y).shape)
    #print(np.array(test_y))

    return maxvalue,mean,std,test_x,test_y

#——————————————————定义网络——————————————————
def lstm(X,weights,biases,input_size,rnn_unit,keep_prob):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']

    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) # reuse = sign
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    m = output_rnn
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    index = tf.range(0, batch_size) * time_step + (time_step - 1) #只取最后的输出
    output = tf.gather(output , index)#按照index取数据
    mm = output
    w_out=weights['out']
    b_out=biases['out']
    pred0=tf.matmul(output,w_out)+b_out
    pred = tf.nn.dropout(pred0, keep_prob)
    return pred,final_states,m,mm

#——————————————————训练模型—————————————————
def train_lstm(data,input_size,output_size,lr,train_time,rnn_unit,weights,biases,train_end,
               batch_size,time_step,kp,train_begin=0):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,output_size])
    keep_prob = tf.placeholder('float')
    batch_index,train_x,train_y=get_train_data(data,batch_size,time_step,train_end,train_begin)
    train_y = np.array(train_y)[:, 1:input_size].tolist()  # 如果输入加上时间维度，这里就需要加上
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    pred,_,m, mm=lstm(X,weights,biases,input_size,rnn_unit,keep_prob)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred, [-1, output_size]) - tf.reshape(Y, [-1, output_size])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #saver.save(sess, './save/MyModel')
        ckpt = tf.train.get_checkpoint_state(save_model_path)  # checkpoint存在的目录
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        else:
            print('No Model')


        #重复训练
        for i in range(train_time):
            for step in range(len(batch_index)-1):
                _,loss_,M,MM=sess.run([train_op,loss,m,mm],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob: kp})
            print(i,loss_)
        saver.save(sess, save_model_path + save_model_name)#保存模型

        #预测
        maxvalue, mean, std, test_x, test_y = get_test_data(data,time_step,train_end - time_step-(predict_step-1))
        test_y = np.array(test_y)[:, 1:input_size].tolist() #如果输入加上时间维度，这里就需要加上
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]],keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        print('test_predict:', np.array(test_predict).shape)
        print('truedata:', np.array(test_y).shape)
        test_predict = np.array(test_predict).reshape(-1, output_size)
        print('test_predict:', test_predict.shape)
        print('test_y:', np.array(test_y).shape)

        test_y = np.array(test_y)
        #for i in range(output_size):
            #test_y[:, i] = test_y[:, i] * std[i+1] + mean[i+1]
            #test_predict[:, i] = test_predict[:, i] * std[i+1] + mean[i+1]
        for i in range(output_size):
            test_y[:, i] = test_y[:, i] * maxvalue[i+1]
            test_predict[:, i] = test_predict[:, i] * maxvalue[i+1]

        # 预测误差
        predict_y = test_predict.reshape(-1, )
        true_y = test_y.reshape(-1, )
        mae = mean_absolute_error(test_predict, test_y)/1024
        mse = mean_squared_error(test_predict, test_y)
        rmse = sqrt(mse)
        r_square = r2_score(test_predict, test_y,multioutput="raw_values")
        r_square1 = r2_score(test_predict, test_y)
        print(('平均绝对误差(mae): %dKB' % mae),
              (mean_absolute_error(test_predict, test_y,multioutput="raw_values")/1024).astype(np.int))
        aa = test_y.sum()/(len(true_y)*1024)
        print('实际值的平均值：%dKB' % aa)
        print('误差百分比：%.2f' % (100*mae/aa)+'%')
        print('均方误差: %d' % mse)
        print('均方根误差: %d' % rmse)
        #print('R_square: %.6f' % r_square)
        print('r2_score:')#越接近1表示预测值和实际值越接近
        print(r_square)
        print(r_square1)

        # 画图表示结果
        fig = plt.figure()
        w1 = []
        w2 = []
        w3 = []
        ax4 = fig.add_subplot(121, projection='3d')
        ax4.set_xlabel('time/min')
        ax4.set_ylabel('area')
        ax4.set_zlabel('read or write/B')
        # ax4.set_yticks([i for i in range(output_size)])
        # ax4.set_yticklabels(['area0', 'area1', 'area2', 'area3', 'area4'])
        # ax4.set_zticks([0, 50*1024*1024, 100*1024*1024, 150*1024*1024, 200*1024*1024, 250*1024*1024, 300*1024*1024])
        # ax4.set_zticklabels(['0', '50', '100', '150', '200', '250', '300'])
        y = [[] for i in range(output_size)]
        z = [[] for i in range(output_size)]
        x = list(range(len(data)))
        y[0] = np.zeros((len(data), 1))
        y[1] = np.ones((len(data), 1))
        Y = [[] for i in range(output_size)]
        Z = [[] for i in range(output_size)]
        X = list(range(train_end, train_end + len(test_predict)))
        Y[0] = np.zeros((len(test_predict), 1))
        Y[1] = np.ones((len(test_predict), 1))
        for i in range(2, output_size):
            y[i] = np.multiply(y[1], [i])
            Y[i] = np.multiply(Y[1], [i])
        for i in range(output_size):
            z[i] = data[:, i+1]
            Z[i] = test_predict[:, i]
        for i in range(output_size):
            ax4.scatter(x, y[i], z[i], color='r', marker='o', s=3)
            #ax4.scatter(X, Y[i], Z[i], color='b', marker='o', s=3)

        ax1 = fig.add_subplot(122)
        ax1.set_title('table')
        #plt.ylim(0,1000000)
        ax1.plot(x, z[1], color='r')
        ax1.plot(X, Z[1], color='b')

        # ax1.plot(list(range(train_end, train_end + len(test_predict))), test_predict[:, 2 * i], color='b',label='predict')
        # ax1.plot(list(range(len(data))), data[:, 2 * i], color='r', label='real')
        plt.legend()
        plt.show()

if __name__=="__main__":
    input_size = 3  # 输入维度
    output_size = 2  # 输出维度
    rnn_unit = 12  # 隐藏层节点
    train_end = 700  # 训练集截取到的位置140/190,320/386,4200/4537
    lr = 0.0004  # 学习率
    train_time = 500  # 所有数据的训练轮次
    batch_size = 30  # 每次训练的一个批次的大小
    time_step = 20  # 前time_step步来预测下一步
    predict_step = 20 #预测predict_step分钟后的负载
    kp = 1  # dropout保留节点的比例
    smooth = 0  # 为1则在时间维度上平滑数据
    filename = 'csv_for_train/read_1_thread_500.csv'
    save_model_path = './save/'  # checkpoint存在的目录
    save_model_name = 'MyModel'    #saver.save(sess, './save/MyModel') 保存模型

    f = open(filename)
    df = pd.read_csv(f)  # 读入数据
    data = df.values
    data = data[1:][:, 0:input_size]
    data = data.astype(np.int)

    if smooth == 1:  # 平滑数据
        newdata = data
        for i in range(2, len(data) - 2):
            for j in range(input_size * 2):
                newdata[i][j] = (data[i - 2][j] + data[i - 1][j] + data[i][j] + data[i + 1][j] + data[i + 2][j]) / 5
        data = newdata

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置

    # random_normal
    # random_uniform
    # truncated_normal
    init_orthogonal = tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32)
    init_glorot_uniform = tf.glorot_uniform_initializer()
    init_glorot_normal = tf.glorot_normal_initializer()

    weights = {
        #'in': tf.get_variable('in', shape=[input_size, rnn_unit], initializer=init_glorot_normal),
        #'out': tf.get_variable('out', shape=[rnn_unit, output_size], initializer=init_glorot_normal)
        'in': tf.Variable(tf.random_uniform([input_size, rnn_unit])),#maxval=0.125
        'out': tf.Variable(tf.random_uniform([rnn_unit, output_size]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    }
    t0 = time.time()
    train_lstm(data,input_size,output_size,lr,train_time,rnn_unit,weights,biases,train_end,batch_size,time_step,kp)
    t1 = time.time()
    print("时间:%.4fs" % (t1 - t0))
