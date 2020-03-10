'''
读取save文件夹下的存储点，这样直接使用训练好的网络来预测。
实时读取数据，每固定时间训练一次，
每次训练后都存储模型，
'''
import numpy as np
import tensorflow as tf
import requests
import json
from sklearn.metrics import *
import time
from os import listdir
from .decode import *
from . import globalvar

kk = 0
history_data =[]
predict_message = []
class tempVar:
    data = {}

#http从pd请求需要的数据
def get_http_data(url):
    data = []

    a = requests.get(url)
    load_dict = json.loads(a.content.decode('utf-8'))

    '''
    #模拟从本地定期读数据
    fdir = '../data/read_1_thread_500/'
    files = listdir(fdir)
    global kk
    f = open(fdir + files[kk], 'r')
    kk += 1
    load_dict = json.load(f)
    f.close()
    nowtime = time.localtime(int(files[kk][0:-4])).tm_min
    '''


    for i in load_dict['regions']:
        temp = []
        temp.append(i['start_key'])
        temp.append(i['end_key'])
        if 'written_bytes' in i.keys():
            temp.append(i['written_bytes'])
        else:
            temp.append(0)
        if 'read_bytes' in i.keys():
            temp.append(i['read_bytes'])
        else:
            temp.append(0)
        data.append(temp)
    time.time()
    data.sort(key=(lambda x: x[0]))
    nowtime =time.localtime(time.time()).tm_min
    return data,nowtime

#把data整合成一行输入,使用全局变量
def get_line_input(output_size, url,tablelist):
    area = [0 for i in range(output_size)]
    num = [0 for i in range(output_size)]
    key_range = [['',''] for i in range(output_size)]
    nowdata,nowtime = get_http_data(url)

    startsign =[0 for i in range(output_size)]
    endsign = [0 for i in range(output_size)]
    for i in range(len(nowdata)):
        key = ''
        if i == 0:  # 因为第一个region的start_key为‘’，所以使用end_key代替
            key = nowdata[i][1]
        else:
            key = nowdata[i][0]
        type, tableid, rowid, indexvalue = decode(key)
        if tablelist.count(tableid) == 1:
            tableindex = tablelist.index(tableid)
            area[tableindex] += nowdata[i][2]+ nowdata[i][3]
            num[tableindex] += 1
            if startsign[tableindex] == 0:
                startsign[tableindex] = 1
                if i != 0: # 第一个region的start_key为''
                    key_range[tableindex][0] = nowdata[i][0]
            #end_key
            if i == len(nowdata)-1:
                pass #最后一个region的end_key为''
            elif endsign[tableindex] == 0:
                _, next_tableid, __, ___ = decode(nowdata[i+1][0])
                if next_tableid != tableid:
                    endsign[tableindex] = 1
                    key_range[tableindex][1] = nowdata[i][1]
    #tempVar.data['key_range'] = key_range
    for i in range(output_size):
        tempVar.data['table_info'][i]['start_key'] = key_range[i][0]
        tempVar.data['table_info'][i]['end_key'] = key_range[i][1]

    now = time.localtime(time.time())  # (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

    global history_data
    history_data[0:-1] = history_data[1:]#丢掉最早的时刻
    #history_data[-1][0] = now.tm_min
    history_data[-1][0] = nowtime

    for i in range(output_size):
        area[i] = area[i] // num[i]
        history_data[-1][i+1] = area[i]

#按照time_step生成预测数据,实际中得不到预测的10分钟后的数据
def get_predict_data(time_step):
    data_test = history_data[-time_step : ]
    print(data_test)
    maxnum = np.max(data_test, axis=0)
    normalized_test_data = data_test / maxnum

    #print('get_test_data x', np.array(normalized_test_data).shape)
    return maxnum, normalized_test_data

#按照batch_size和time_step生成训练数据,标签是10分钟之后的数据
def get_train_data(batch_size,time_step,predict_step):
    maxnum = np.max(history_data, axis=0)
    normalized_train_data = history_data  / maxnum

    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step - (predict_step - 1)):
        x = normalized_train_data[i:i + time_step]
        y = normalized_train_data[i + time_step + (predict_step - 1)]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return train_x, train_y

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

#——————————————————模型—————————————————
def train_lstm(input_size,output_size,lr,rnn_unit,weights,biases,batch_size,time_step,kp,url,
               err_eval_step,predict_step,tablelist,save_model_path,save_model_name):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,output_size])
    keep_prob = tf.placeholder('float')
    pred,_,m, mm=lstm(X,weights,biases,input_size,rnn_unit,keep_prob)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred, [-1, output_size]) - tf.reshape(Y, [-1, output_size])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(save_model_path)  # checkpoint存在的目录
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
            print("Model restored...")
        else:
            print('No Model')


        label = 0  # 当前的序号
        train_num = 0 #累积多久训练一次
        r_square = [0 for i in range(output_size)]
        r_square_total = 0
        global predict_message
        while True:#每个预测就是一次循环，每隔一段时间就训练一次，存储模型
            # 预测
            if label > 1000: #值没有意义，只是运行一段时间后不需要用到label,
                pass
            else:
                label += 1
            get_line_input(output_size, url,tablelist)
            print('label:',label)
            if label >= time_step + predict_step:
                train_num += 1
            if label >= time_step:
                maxnum, test_x = get_predict_data(time_step)
                maxvalue = np.max(test_x, axis=0)[1:]
                minvalue = np.min(test_x, axis=0)[1:]
                prob = sess.run(pred, feed_dict={X: [test_x], keep_prob: 1})
                predict = prob.reshape((-1))
                predict_message[0:-1] = predict_message[1:]  # 丢掉最早的时刻
                for i in range(output_size):
                    predict[i] = predict[i] * maxnum[i+1] # i+1是因为输出少一维时间
                    maxvalue[i] = maxvalue[i] * maxnum[i + 1]
                    minvalue[i] = minvalue[i] * maxnum[i + 1]
                    if predict[i]<0:
                        predict[i] = 0
                    predict_message[-1][i] = predict[i]
                if label >= time_step+2*predict_step+2*err_eval_step-1:#计算之前预测的平均相对误差
                    r_square = [0 for i in range(output_size)]
                    r_square_total = 0
                    for i in range(2 * err_eval_step + 1):  # 在时间+- x min的实际曲线上计算误差，取最小值
                        last_predict = predict_message[err_eval_step:err_eval_step + predict_step]
                        beginindex = batch_size + time_step - 1 - i
                        last_y = history_data[beginindex:beginindex + predict_step, 1:]
                        mae = mean_absolute_error(last_predict, last_y) / 1024
                        print(('平均绝对误差(mae): %dKB' % mae),
                              (mean_absolute_error(last_predict,last_y,multioutput="raw_values")/1024).astype(np.int))
                        aa = last_y.sum() / (predict_step * output_size * 1024)
                        print('实际值的平均值：%dKB' % aa)
                        print('误差百分比：%.2f' % (100 * mae / aa) + '%')
                        r_square_temp = r2_score(last_predict, last_y, multioutput="raw_values")
                        r_square_total_temp = r2_score(last_predict, last_y)
                        print(r_square_total_temp, r_square_temp)
                        if r_square_total_temp > r_square_total:
                            r_square = r_square_temp.tolist()
                            r_square_total = r_square_total_temp

                # plt.pause(0.5)
                # ax.plot(list(range(label,label + predict_step)), predict_message[:,0], color='b')
                #更新要发送的数据,其中每个表的key范围在get_line_input中更新
                tempVar.data['time'] = int(time.time())
                tempVar.data['predict_step'] = predict_step
                tempVar.data['history_r2_score_total'] = r_square_total
                tempVar.data['table_num'] = output_size
                for i in range(output_size):
                    tempVar.data['table_info'][i]['max_value'] =maxvalue[i]
                    tempVar.data['table_info'][i]['min_value'] =minvalue[i]
                    tempVar.data['table_info'][i]['predict'] =predict_message[-predict_step:][:,i].tolist()
                    tempVar.data['table_info'][i]['history_r2_score'] =r_square[i]

            #训练
            if label>=batch_size+time_step+predict_step-1 and train_num == batch_size:
                # batch_size=30, timestep=30, predict_step=10,第一次训练[0:30]->[39]-[29:59]->[68],这时label=69
                train_x, train_y = get_train_data(batch_size, time_step,predict_step)#x:(batch_size, time_step, input_size)y:(batch_size, output_size)
                train_y = np.array(train_y)[:, 1:input_size].tolist()  # 如果输入加上时间维度，这里就需要加上
                _, loss_, M, MM = sess.run([train_op, loss, m, mm],feed_dict={X: train_x,Y: train_y,keep_prob: kp})
                print('label,loss: ',label, loss_)
                train_num = 0
                saver.save(sess, save_model_path + save_model_name)

            time.sleep(60) #每隔60s循环一次，实际上因为程序运行所以超过一分钟


def start_predict():
    input_size = 3  # 输入维度
    output_size = 2  # 输出维度
    rnn_unit = 12 # 隐藏层节点
    #train_end = 200  # 训练集截取到的位置
    lr = 0.0004  # 学习率
    #train_time = 100  # 所有数据的训练轮次
    batch_size = 30  # 每次训练的一个批次的大小
    time_step = 20  # 前time_step步来预测下一步
    predict_step = 20  #预测predict_step分钟后的负载
    err_eval_step = 2  #把预测曲线和前后几分钟的实际曲线相比较，取误差最小的值当成预测误差。
    kp = 1  # dropout保留节点的比例
    tempVar.data = globalvar.get_demo_value()
    tempVar.data['table_info'] =[{} for i in range(output_size)]
    # 'http://192.168.1.128:2379/pd/api/v1/regions'
    url = 'http://10.233.57.252:2379/pd/api/v1/regions'
    tablelist = [45, 47]  # 4个表id [95, 97, 113, 114]
    save_model_path = './save/'  # checkpoint存在的目录
    save_model_name = 'MyModel'  # saver.save(sess, './save/MyModel') 保存模型

    global history_data
    global predict_message
    history_data = np.zeros((batch_size + time_step + predict_step - 1, input_size))
    predict_message = np.zeros((2 * predict_step + 2 * err_eval_step, output_size))
    # predict_message长度超过预测的步数，即里面也包括过时的预测信息，用于计算预测准确度

    # ——————————————————定义神经网络变量——————————————————
    #如果是加载已训练好的模型，w和b应该是相同的形状
    weights = {
        'in': tf.Variable(tf.random_uniform([input_size, rnn_unit])),#maxval=0.125
        'out': tf.Variable(tf.random_uniform([rnn_unit, output_size]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    }
    train_lstm(input_size,output_size,lr,rnn_unit,weights,biases,batch_size,
               time_step,kp,url,err_eval_step,predict_step,tablelist,save_model_path,save_model_name)