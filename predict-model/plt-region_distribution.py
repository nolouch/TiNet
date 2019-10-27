'''
看某一时刻region中的读写分布情况,结果存在sun.csv中
'''
import json
import numpy as np
import matplotlib.pyplot as plt
import csv

data = []
filename = 'data/hack-data/data/20191027-00-01.json'
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

    w = open("sun.csv", 'w', newline='')
    csv_write = csv.writer(w)
    for i in range(len(data)):
        print(i,data[i])
        csv_write.writerow(data[i])
    w.close()

    #print(data[0])
    #print(data[-1])

    x = list(range(len(data)))

    y = np.array(data)[:,2]
    y = list(map(int, y.reshape(-1,).tolist()))


    plt.figure()
    plt.xlabel('region')
    plt.ylabel('written_bytes')
    #plt.ylim(0, 10000000)
    plt.scatter(np.array(x), y, c='r', marker='o', s=3)
    plt.show()
