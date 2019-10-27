'''
在linux系统下通过命令行调用mok工具解码，把数据按表id、行id等分组
结果存在clustering_test_data.csv中，用于clustering_test.py的聚类
或者调用decode.py函数解码，更快
'''
import os
import json
import numpy as np
import csv
from decode import *
import matplotlib.pyplot as plt

filename = './data/single-update/20191026-09-30.json'   #要解码的文件
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
        data.append(temp)

    data.sort(key=(lambda x: x[0]))
    print(len(data))
    print(data[0])
    print(data[-1])

    x = list(range(len(data)))
    y = np.array(data)[:,2]
    y = list(map(int, y.reshape(-1,).tolist()))


    w = open("clustering_test_data.csv", 'w', newline='')
    csv_write = csv.writer(w)
    csv_write.writerow(["index", "tableid", "rowid/indexid", "indexvalue", "written_bytes"])
    for i in range(len(data)):
        key = ''
        if i == 0:#因为第一个region的start_key为‘’，所以使用end_key代替
            key = data[i][1]
        else:
            key = data[i][0]
        if i% 1000 == 0:
            print(i)
        type,tableid,rowid,indexvalue =decode(key)
        csv_write.writerow([i,tableid,rowid,indexvalue, data[i][2]])
        '''
        decode = os.popen('mok '+ key)
        code = decode.readlines()
        for j in range(len(code)):
            if(code[j].find('table prefix')!=-1):
                tableid = code[j + 1].split(':')[1].strip('\n')
                csv_write.writerow([i,1,int(tableid),2,1,data[i][2]])
                break
            elif(code[j].find('table index key')!=-1):
                tableid = code[j + 1].split(':')[1].strip('\n')
                indexid = code[j + 2].split(':')[1].strip('\n')
                indexline = code[j + 3][code[j + 3].find('user')+4:]
                indexvalue = indexline.split('\\')[0]
                #print(i,2, tableid, indexid, indexvalue, data[i][2])
                csv_write.writerow([i,2,int(tableid),int(indexid),int(indexvalue),data[i][2]])
                #csv_write.writerow([i, 2, int(tableid), int(indexid), 2, data[i][2]])
                break
            elif(code[j].find('table row key')!=-1):
                tableid = code[j + 1].split(':')[1].strip('\n')
                rowid = code[j + 2].split(':')[1].strip('\n')
                #print(i,3, tableid, rowid, data[i][2])
                csv_write.writerow([i,3,int(tableid),int(rowid),3,data[i][2]])
                break
        '''

    w.close()






