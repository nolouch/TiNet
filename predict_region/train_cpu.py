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

from train import get_test_data,get_train_data,lstm

def read_cpu(filename):
    with open(filename, 'rb') as f:
        df = pd.read_csv(f,sep=';')  # 读入数据
        cpudata = df.values
        cpudata = cpudata[:, 1:]
        cpudata = cpudata.astype(np.float)
        x = np.sum(cpudata, axis=1)[:,np.newaxis]
        print('cpudata.shape:', cpudata.shape)
        print('x.shape:',x.shape)
        return x


#——————————————————训练模型—————————————————
def train_lstm(data,input_size,output_size,lr,train_time,rnn_unit,weights,biases,train_end,
               batch_size,time_step,kp,train_begin=0):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,output_size])
    keep_prob = tf.placeholder('float')
    batch_index,train_x,train_y=get_train_data(data,batch_size,time_step,train_end,predict_step,train_begin)
    #train_y = np.array(train_y)[:, 1:input_size].tolist()  # 如果输入加上时间维度，这里就需要加上
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
        maxvalue, mean, std, test_x, test_y = get_test_data(data,time_step,train_end - time_step-(predict_step-1),predict_step)
        #test_y = np.array(test_y)[:, 1:input_size].tolist() #如果输入加上时间维度，这里就需要加上
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
            test_y[:, i] = test_y[:, i] * maxvalue[i]
            test_predict[:, i] = test_predict[:, i] * maxvalue[i]

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
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('time/min')
        ax1.set_ylabel('cpu usage')
        ax1.set_title('CPU')
        #ax2 = fig.add_subplot(122)
        #ax2.set_xlabel('time/min')
        #ax2.set_ylabel('area')
        #ax2.set_title('table2')
        x = list(range(len(data)))
        X = list(range(train_end, train_end + len(test_predict)))
        y = [[] for i in range(output_size)]
        Y = [[] for i in range(output_size)]
        for i in range(output_size):
            y[i] = data[:, i]
            Y[i] = test_predict[:, i]
        ax1.plot(x, y[0], color='b', label='real')
        ax1.plot(X, Y[0], color='r', label='predict')
        #ax2.plot(x, y[1], color='r', label='real')
        #ax2.plot(X, Y[1], color='b', label='predict')
        plt.legend()
        plt.show()

if __name__=="__main__":
    input_size = 1  # 输入维度
    output_size = 1  # 输出维度
    rnn_unit = 12  # 隐藏层节点
    lr = 0.0004  # 学习率
    batch_size = 30  # 每次训练的一个批次的大小
    time_step = 20  # 前time_step步来预测下一步
    predict_step = 5 #预测predict_step分钟后的负载
    kp = 1  # dropout保留节点的比例
    smooth = 0  # 为1则在时间维度上平滑数据
    train_time = 50  # 所有数据的训练轮次
    train_end = 1200  # 训练集截取到的位置140/190,320/386,4200/4537
    filename = 'csv_for_train/grafana_data_export_4-18_60.csv'
    save_model_path = './save/predict_cpu_4-18_60/'  # checkpoint存在的目录
    save_model_name = 'MyModel'    #saver.save(sess, './save/MyModel') 保存模型

    data = read_cpu(filename)

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
