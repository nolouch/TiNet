'''
1：按照表里多少行来划分区域预测，记录每个区的key范围，之后每个区的region数可能变，
结果存在lstm4_9.9_data.csv中（记录的是区域里所有region负载的平均值）
2：每个时间点的各区域范围会变，比如每个时间点的某个区域是表的前10000行对应的region，
最后预测出高负载的话，使用最后一个时间点区域的范围边界。
'''
import json
import csv
from os import listdir

dirname = 'data/multiple-update/data/'
area_num = 10
area_size = 20
startkey = ['' for i in range(area_num)]
endkey = ['' for i in range(area_num)]

files = listdir(dirname)
print('len(files): ',len(files))

w = open("lstm_multiple-update_data.csv", 'w', newline='')
csv_write = csv.writer(w)
csv_head = ['area'+str(i) for i in range(area_num)]+['num'+str(i) for i in range(area_num)]
print(csv_head)
csv_write.writerow(csv_head)

for k in range(len(files)):#读所有的文件
    data = []
    with open(dirname + files[k], 'r') as f:
        lines = f.readlines()
        if(len(lines)==1):
            continue #忽略这一文件
        f.seek(0, 0)
        load_dict = json.load(f)
        for i in load_dict['regions']:
            temp = []
            temp.append(i['start_key'])
            temp.append(i['end_key'])
            #temp.append(i["approximate_size"])
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

    if k == 0:
        for i in range(area_num-1):
            startkey[i] = data[i*area_size][0]
            endkey[i] = data[(i+1)*area_size-1][1]
        startkey[area_num-1] = data[(area_num-1) * area_size][0]
        endkey[area_num-1] = data[-1][1]
        csv_write.writerow(startkey + endkey)

    area = [0 for i in range(area_num)]
    num = [0 for i in range(area_num)]
    for i in range(len(data)):
        if(data[i][0] >= startkey[area_num-1]):
            area[area_num-1] += data[i][2]
            num[area_num-1] += 1
        else:
            for j in range(area_num-1):
                if(data[i][0] >= startkey[j])and(data[i][0] < endkey[j]):#考虑最开始的‘’
                    area[j] += data[i][2]
                    num[j] += 1

    for i in range(area_num):
        area[i] = area[i] // num[i]

    print(files[k], area, num)
    csv_write.writerow(area + num)
w.close()
