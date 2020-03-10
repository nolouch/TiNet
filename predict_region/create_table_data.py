'''
创建数据，
对10.30-update-10min数据的测试，按表来分区，分成4个表的负载的平均值。
输入除了表的读加写比特，再加上一个时间，值域为0-59。
每读一个文件，先全部读出来，再全部解码，再计算4个表的written_bytes+read_bytes的平均值
（这里测试就直接按表id来选,45/47），
再看中间是否差时间，差的话，先在数据集后补上应该有的时间，再加上这个时刻的。
'''
import json
import csv
import time
from os import listdir
from decode import *

dirname = 'data/read_1_thread_500/'
savename = "csv_for_train/read_1_thread_500.csv"
area_num = 2
tablelist = [45,47]  #[95,97,113,114] [95,97,101,103,113,114]，[43,45],[45,47]

files = listdir(dirname)
print('len(files): ',len(files))

w = open(savename, 'w', newline='')
csv_write = csv.writer(w)
csv_head = ['time']+['area'+str(i) for i in range(area_num)]+['num'+str(i) for i in range(area_num)]
print(csv_head)
csv_write.writerow(csv_head)

lastdata = []
for k in range(len(files)):#读所有的文件，   t-data-0从第200个文件开始
    data = []
    with open(dirname + files[k], 'r') as f:
        #lines = f.readlines()
        #if(len(lines)<=1):
            #continue #忽略这一文件
        #f.seek(0, 0)
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

    area = [0 for i in range(area_num)]
    num = [0 for i in range(area_num)]
    hot_data_index = [0 for i in range(area_num)]
    for i in range(len(data)):
        key = ''
        if i == 0:  # 因为第一个region的start_key为‘’，所以使用end_key代替
            key = data[i][1]
        else:
            key = data[i][0]
        type, tableid, rowid, indexvalue = decode(key)
        if (tablelist.count(tableid)):#把数据的读写都加上
            area[tablelist.index(tableid)] = area[tablelist.index(tableid)] + data[i][2] + data[i][3]
            num[tablelist.index(tableid)] += 1

    for i in range(area_num):
        area[i] = area[i] // num[i]  #注释了是总负载，不注释是平均负载
        #hot_data_index[i] = find_hot_data(tablelist[i])

    #补上缺少的时间，缺少的直接使用前一刻的数据
    #nowtime = int(files[k][-7:-5])
    print(k,int(files[k][0:-4]))
    nowtime = time.localtime(int(files[k][0:-4])).tm_min
    if k==0:
        print(files[k], nowtime, area)
        lastdata = [nowtime] + area + num
        csv_write.writerow(lastdata)
    else:
        for i in range((nowtime + 59 - lastdata[0]) % 60):
            print(files[k], (lastdata[0]+i+1)% 60, lastdata[1:])
            csv_write.writerow([(lastdata[0]+i+1)% 60] + lastdata[1:])
        print(files[k], nowtime, area, num)
        lastdata = [nowtime] + area  + num
        csv_write.writerow(lastdata)
w.close()