'''
读取save文件夹下的存储点，这样直接使用训练好的网络来预测。
实时读取数据，每固定时间训练一次，
每次训练后都存储模型，
'''
import numpy as np
import tensorflow as tf
import requests
import json
import yaml
from sklearn.metrics import *
import time
from os import listdir
from .decode import *
from . import globalvar
from .predict import lstm
from math import ceil
import random
import ast
import subprocess


minReplicas = 4
maxReplicas = 5
scaleIntervalmins = 10
averageUtilization = 3 #3/8


kk = 0
history_data =[]
class tempVar:
    data = {}

def yaml_to_dict(yaml_path):
    with open(yaml_path, "r") as test_file:
        generate_dict = yaml.load(test_file, Loader=yaml.FullLoader)  # 先将yaml转换为dict格式
        # generate_json = json.dumps(generate_dict,sort_keys=False,indent=4,separators=(',',': '))
        init_tikv_replicas = int(generate_dict['spec']['tikv']['replicas'])
        cpustr = generate_dict['spec']['tikv']['limits']['cpu']
        if cpustr[-1] == 'm':
            cpurequest = float(cpustr[0:-1]) / 1000
        else:
            cpurequest = float(cpustr)
        print('cpurequest:', cpurequest)
        return cpurequest, init_tikv_replicas



def fetch_tikv_cpu_usage(prome_addr, start, end, step=30):
    r = requests.get(
        'http://%s/api/v1/query_range?query=sum(rate(tikv_thread_cpu_seconds_total[1m])) by (instance)&start=%s&end=%s&step=%s' % (
            prome_addr, start, end, step))
    res = r.json()
    if res['status'] == 'error':
        raise Exception(
            'an error occurred when fetching tikv cpu usage: errorType={}: {}'.format(res['errorType'], res['error']))
    return res['data']['result']

#从Prometheus获取cpu数据,把data整合成一行输入,使用全局变量
def get_cpu_input(output_size, url, interval):
    global history_data
    sum = 0

    '''
    cpu_list = []
    if interval == 0:
        with open('cpu_data/example1.txt', 'r') as f:
            load_dict = f.readlines()
            for node in load_dict:
                node = node.strip('\n')
                node = ast.literal_eval(node)
                cpu_list.append(node)
        for ins in cpu_list:
            sum += float(ins['values'][0][1])
        history_data[0:-1] = history_data[1:]  # 丢掉最早的时刻
        history_data[-1][0] = sum
    else:
        with open('cpu_data/example2.txt', 'r') as f:
            load_dict = f.readlines()
            for node in load_dict:
                node = node.strip('\n')
                node = ast.literal_eval(node)
                cpu_list.append(node)
        for i in range(-interval,0):
            for ins in cpu_list:
                history_data[i][0] += float(ins['values'][i][1])
    '''

    prome_addr = '10.233.18.170:9090'
    step = 60
    end = str(int(time.time()))

    if interval == 0:
        start = str(int(time.time()))
        cpudata = fetch_tikv_cpu_usage(prome_addr, start, end, step)
        for ins in cpudata:
            sum += float(ins['values'][0][1])
        history_data[0:-1] = history_data[1:]  # 丢掉最早的时刻
        history_data[-1][0] = sum
    else:
        start = str(int(time.time())-60*(interval-1))#隔1min获取的话是获取interval个数据
        cpudata = fetch_tikv_cpu_usage(prome_addr, start, end, step)
        for i in range(-interval, 0):
            for ins in cpudata:
                history_data[i][0] += float(ins['values'][i][1])

#按照time_step生成预测数据,实际中得不到预测的10分钟后的数据
def get_predict_data(time_step):
    data_test = history_data[-time_step : ]
    #print(data_test)
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


#——————————————————模型—————————————————
def train_lstm(input_size,output_size,lr,rnn_unit,weights,biases,batch_size,time_step,kp,url,
               predict_step,save_model_path,save_model_name,init_tikv_replicas,yaml_path):
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

        last_scale_time = -scaleIntervalmins
        current_replicas = init_tikv_replicas
        label = 0  # 当前的序号
        train_num = 0 #累积多久训练一次
        while True:#每个预测就是一次循环，每隔一段时间就训练一次，存储模型
            # 预测
            if label > 1000: #值没有意义，只是运行一段时间后不需要用到label,
                pass
            else:
                label += 1
            if label == 1:
                get_cpu_input(output_size, url, time_step)
            else:
                get_cpu_input(output_size, url, 0)

            #预测
            maxnum, test_x = get_predict_data(time_step)
            prob = sess.run(pred, feed_dict={X: [test_x], keep_prob: 1})
            predict = prob.reshape((-1))
            for i in range(output_size):
                predict[i] = predict[i] * maxnum[i]

                # 计算期望副本数
                pre_replicas = ceil(predict[i] / averageUtilization)
                if pre_replicas > maxReplicas:
                    pre_replicas = maxReplicas
                if pre_replicas < minReplicas:
                    pre_replicas = minReplicas
                tempVar.data['tikv_replicas'] = pre_replicas
                print("label_cpu:%d, time:%s, predict_step:%d, predict_tikv_cpu_usage:%f, predict_tikv_replicas:%d"
                      % (label, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), predict_step, predict[i], pre_replicas))
                # 判断是否调度，考虑调度间隔时间等
                if (label - last_scale_time >= scaleIntervalmins) and (pre_replicas != current_replicas) :
                    #修改yaml配置文件
                    generate_dict = {}
                    with open(yaml_path, "r") as yaml_file:
                        generate_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)  # 先将yaml转换为dict格式
                    with open(yaml_path, "w") as yaml_file:
                        generate_dict['spec']['tikv']['replicas'] = pre_replicas
                        yaml.dump(generate_dict, yaml_file)
                    #执行shell命令
                    exitcode, output = subprocess.getstatusoutput("kubectl apply -f %s -n pd-team-s2" % yaml_path)
                    print("exitcode: ", exitcode)
                    print("output: ", output)
                    if exitcode == 0:
                        print("Execute scaling command, tikv_replicas:%d -> %d" % (current_replicas, pre_replicas))
                        last_scale_time = label
                        current_replicas = pre_replicas


            if label >= predict_step:
                train_num += 1
            #训练
            if label>=batch_size+predict_step-1 and train_num == batch_size:
                # batch_size=30, timestep=30, predict_step=10,第一次训练[0:30]->[39]-[29:59]->[68],这时label=69
                train_x, train_y = get_train_data(batch_size, time_step,predict_step)#x:(batch_size, time_step, input_size)y:(batch_size, output_size)
                #train_y = np.array(train_y)[:, 1:input_size].tolist()  # 如果输入加上时间维度，这里就需要加上
                _, loss_, M, MM = sess.run([train_op, loss, m, mm],feed_dict={X: train_x,Y: train_y,keep_prob: kp})
                print('label_cpu,loss: ',label, loss_)
                train_num = 0
                saver.save(sess, save_model_path + save_model_name)

            time.sleep(60) #每隔60s循环一次，实际上因为程序运行所以超过一分钟


def start_predict_cpu():
    input_size = 1  # 输入维度
    output_size = 1  # 输出维度
    rnn_unit = 12 # 隐藏层节点
    lr = 0.0004  # 学习率
    batch_size = 30  # 每次训练的一个批次的大小
    time_step = 20  # 前time_step步来预测下一步
    predict_step = 5  #预测predict_step分钟后的负载
    kp = 1  # dropout保留节点的比例
    tempVar.data = globalvar.get_demo_value()
    url = 'http://10.233.22.61:2379/pd/api/v1/regions'
    save_model_path = './save/predict_cpu_4-18_60/'  # checkpoint存在的目录
    save_model_name = 'MyModel'  # saver.save(sess, './save/MyModel') 保存模型

    yaml_path = '/data2/hust_tmp/cluster/tidb-cluster.yaml' #读配置文件，怎么才不需要设置成绝对路径
    cpurequest, init_tikv_replicas = yaml_to_dict(yaml_path)#读配置的limits，和初始的tikv的replicas



    global history_data
    history_data = np.zeros((batch_size + time_step + predict_step - 1, input_size))


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
               time_step,kp,url,predict_step,save_model_path,save_model_name,init_tikv_replicas,yaml_path)
