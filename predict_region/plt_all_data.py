'''
画时间，region，written_bytes的三维图，
其中region就是按照数量来排，可能t0时刻有800个点，t1时刻有900个点。（不一定同一列的点一直表示的是同一个region）
例如9.18的数据是1158个，每个时间6000多个region，每隔10分钟，每20个region作为一个点
'''
import json
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

smooth = 0 #为1则平滑某一时刻的负载
timestep = 1
regionstep = 1
RorW = 3 #2为写，3为读
dirname = 'data/update_thread_100/'

files = listdir(dirname)
print('len(files): ',len(files))

#writename = 'test3_data.csv'
#w = open(writename, 'w', newline='')
#csv_write = csv.writer(w)
#csv_head = ['area'+str(i) for i in range(area_num)]
#csv_write.writerow(csv_head)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('time/min')
ax.set_ylabel('region')
ax.set_zlabel('write/B')
t = 0
k = 0
X = []
Y = []
Z = []

while k < len(files):#读所有的文件len(files)
    data = []
    with open(dirname + files[k], 'r') as f:
        load_dict = json.load(f)
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

        data.sort(key=(lambda x: x[0]))
        print(k,len(data))

        # 平滑数据
        if smooth == 1:
            newdata = data
            for i in range(1, len(data) - 1):
                newdata[i][RorW] = (data[i - 1][RorW] + data[i][RorW] + data[i + 1][RorW]) / 3
            data = newdata

        written_sum = 0
        for i in range(len(data)):
            written_sum += data[i][RorW]
            if (i+1)% regionstep == 0:
                if written_sum >=0:   # written_sum  < 10000000 or >=0
                    X.append(k)
                    Y.append(((i+1)//regionstep)-1)
                    Z.append(written_sum)
                written_sum = 0

    k += timestep

print(len(X),len(Y),len(Z))
#ax.scatter(X, Y, Z, color='r', marker='o', s=3)
ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()


