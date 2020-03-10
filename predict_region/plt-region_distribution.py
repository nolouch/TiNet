'''
看某一时刻region中的读写分布情况,结果存在sun.csv中
'''
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from decode import *

filename = 'data/region/1583607128.log'
RorW = 3 #2为写，3为读

data = []
with open(filename, 'r') as f:
    load_dict = json.load(f)
    for i in load_dict['regions']:
        temp =[]
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
        if 'id' in i.keys():
            temp.append(i['id'])
        data.append(temp)

    data.sort(key=(lambda x: x[0]))
    print(len(data))

    w = open("sun.csv", 'w', newline='')
    csv_write = csv.writer(w)
    for i in range(len(data)):
        print(i,data[i])
        csv_write.writerow(data[i])
    w.close()

    #print(data[0])
    #print(data[-1])
    plt.figure()
    mok_data = []
    for i in range(len(data)):
        key = ''
        if i == 0:  # 因为第一个region的start_key为‘’，所以使用end_key代替
            key = data[i][1]
        else:
            key = data[i][0]
        type, tableid, rowid, indexvalue = decode(key)
        mok_data.append([i, type, tableid, rowid, indexvalue, data[i][RorW]])
        if type==2:#row
            plt.scatter(i, data[i][RorW], c='r', marker='o', s=20)
        if type==3 or type==1 or type==0:#index
            plt.scatter(i, data[i][RorW], c='b', marker='o', s=20)
    mok_data = np.array(mok_data)
    for i in mok_data:
        if i[1] != 3:    # 2是row，3是index
            i[5] = 0

    x = list(range(len(data)))
    #y = np.array(data)[:, 2]
    y = mok_data[:, 5]
    y = list(map(int, y.reshape(-1, ).tolist()))

    plt.xlabel('region list')
    plt.ylabel('read/B')
    plt.title('region distribution')
    #plt.ylim(0, 10000000)
    #plt.scatter(np.array(x), y, c='r', marker='o', s=20)
    plt.show()

