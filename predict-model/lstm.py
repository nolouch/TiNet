'''
使用create_lstm_data.py中生成的文件来预测  多维单步  单个热点预测
比如1634个文件，前1400用于训练，后234个用于测试，实际上由于time_step为10，所以实际上用于训练和测试的是1390和234个。
目前不能跑，其他文件从这调函数
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

#获取训练集
def get_train_data(data,batch_size,time_step,train_end,train_begin=0):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    maxvalue = np.max(data_train,axis=0)
    normalized_train_data = data_train/np.max(data_train,axis=0)

    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step]
       y = normalized_train_data[i+time_step]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(data,time_step,test_begin):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    #normalized_test_data=(data_test-mean)/std  #标准化
    maxvalue = np.max(data_test, axis=0)
    print(maxvalue)
    maxvalue += 1
    print(maxvalue)
    normalized_test_data = data_test / maxvalue
    #size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]
    for i in range(len(normalized_test_data)-time_step):
       x=normalized_test_data[i:i+time_step]
       y=normalized_test_data[i+time_step]
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
def train_lstm(data,input_size,output_size,lr,train_time,rnn_unit,weights,biases,kp,
               batch_size=20,time_step=10,train_begin=0,train_end=100):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,output_size])
    keep_prob = tf.placeholder('float')
    batch_index,train_x,train_y=get_train_data(data,batch_size,time_step,train_begin,train_end)
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    pred,_,m, mm=lstm(X,weights,biases,input_size,rnn_unit,keep_prob)
    #损失函数
    #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1,3]) - Y))
    loss=tf.reduce_mean(tf.square(tf.reshape(pred, [-1, output_size]) - tf.reshape(Y, [-1, output_size])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(train_time):
            for step in range(len(batch_index)-1):
                _,loss_,M,MM=sess.run([train_op,loss,m,mm],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob: kp})
                #m, mm= sess.run([ m,mm],feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],Y: train_y[batch_index[step]:batch_index[step + 1]]})
                #print('M ',M.shape)
                #print('MM ', MM.shape)
            print(i,loss_)

        #预测
        mean, std, test_x, test_y = get_test_data(data,time_step,train_end - time_step)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]],keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        print('test_predict:', np.array(test_predict).shape)
        test_predict = np.array(test_predict).reshape(-1, output_size)
        print('test_predict:', test_predict.shape)
        print('test_y:', np.array(test_y).shape)

        test_y = np.array(test_y)
        for i in range(output_size):
            test_y[:, i] = test_y[:, i] * std[i] + mean[i]
            test_predict[:, i] = test_predict[:, i] * std[i] + mean[i]

        # 画折线图表示结果
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.set_title('num')
        ax1.plot(list(range(100,100+len(test_predict))), test_predict[:,0], color='b',label = 'predict')
        ax1.plot(list(range(len(data))), data[:,0], color='r',label = 'real')
        plt.legend()

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.set_title('key')
        ax2.plot(list(range(100, 100 + len(test_predict))), test_predict[:, 1], color='b',label='predict')
        ax2.plot(list(range(len(data))), data[:, 1], color='r',label='real')
        plt.legend()

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.set_title('written_bytes')
        ax3.plot(list(range(100, 100 + len(test_predict))), test_predict[:, 2], color='b',label='predict')
        ax3.plot(list(range(len(data))), data[:, 2], color='r',label='real')
        plt.legend()

        ax4 = fig.add_subplot(144, projection='3d')
        ax4.set_xlabel('time')
        ax4.set_ylabel('key')
        ax4.set_zlabel('written_bytes')

        x = list(range(len(data)))
        y = data[:, 1]
        z = data[:, 2]
        #ax4.scatter(x, y, z)
        x1 = list(range(100, 100 + len(test_predict)))
        y1 = test_predict[:, 1]
        z1 = test_predict[:, 2]
        ax4.scatter(x, y, z, color='r',marker='o')
        ax4.scatter(x1, y1, z1, color='r',marker='x')
        #ax4.set_yticklabels(['one', 'two', 'three', 'four'], rotation=0, fontsize='small')
        ax4.set_yticks([1000000, 1500000, 2000000, 2500000])
        ax4.set_yticklabels(['1M', '1.5M', '2M', '2.5M'])
        ax4.set_zticks([0, 2000000, 4000000, 6000000])
        ax4.set_zticklabels(['0', '2M', '4M', '6M'])
        plt.show()


if __name__=="__main__":
    # 定义常量
    rnn_unit = 8  # 隐藏层节点
    input_size = 3  # 输入维度
    output_size = 3  # 输出维度
    lr = 0.0007  # 学习率
    train_time = 500  # 训练轮次
    kp = 1

    # 导入数据
    f = open('lstm_data.csv')
    df = pd.read_csv(f)  # 读入数据
    data = df.values
    data = data.astype(np.int)
    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置

    # random_normal
    # random_uniform
    # truncated_normal
    init_orthogonal = tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32)
    init_glorot_uniform = tf.glorot_uniform_initializer()
    init_glorot_normal = tf.glorot_normal_initializer()

    weights = {
        # 'in': tf.get_variable('in', shape=[input_size, rnn_unit], initializer=init_glorot_normal),
        # 'out': tf.get_variable('out', shape=[rnn_unit, output_size], initializer=init_glorot_normal)
        'in': tf.Variable(tf.random_uniform([input_size, rnn_unit], maxval=0.125)),
        'out': tf.Variable(tf.random_uniform([rnn_unit, output_size]))
    }

    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    }

    train_lstm(data,input_size,output_size,lr,train_time,rnn_unit,weights,biases,kp)



