'''
读取mok.py存的clustering_test_data.csv
按照tableid,rowid/indexid, indexvalue,written_bytes来聚类
'''
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#colors = ['b', 'c', 'r', 'k', 'm', 'g', 'y', 'navy', 'coral','purple','b', 'c', 'r', 'k', 'm', 'g', 'y', 'navy', 'coral','purple']  # 颜色列表
colors = [
 '#F0F8FF',
     '#DB7093',
        '#FFEFD5',
           '#FFDAB9',
              '#CD853F',
                 '#FFC0CB',
               '#DDA0DD',
          '#B0E0E6',
             '#800080',
              '#FF0000',
        '#BC8F8F',
         '#4169E1',
         '#8B4513',
           '#FA8072',
        '#FAA460',
         '#2E8B57',
          '#FFF5EE',
          '#A0522D',
          '#C0C0C0',
         '#87CEEB',
          '#6A5ACD',
        '#708090',
         '#FFFAFA',
        '#00FF7F',
           '#4682B4',
           '#D2B48C',
             '#008080',
           '#D8BFD8',
           '#FF6347',
           '#40E0D0',
             '#EE82EE',
               '#F5DEB3',
            '#FFFFFF',
        '#F5F5F5',
            '#FFFF00',
        '#9ACD32']

# 导入数据
f = open('clustering_test_data.csv')#index,type,tableid,rowid/indexid,indexvalue,written_bytes
df = pd.read_csv(f)  # 读入数据
data = df.values
data = data.astype(np.int)
test_data = data[:,1:]
print('data.shape:',test_data.shape)

#标准化
mean=np.mean(test_data,axis=0)
std=np.std(test_data,axis=0)
print('mean std:')
print(mean)
print(std)
#各维度分权重
std[0] = std[0]/100 #tableid
std[1] = std[1]/7 #rowid/indexid
std[2] = std[2]/1 #indexvalue
std[3] = std[3]/4 #written_bytes
print(std)
normalized_test_data=(test_data-mean)/std

types_num = 30 #总共聚类数量
want_num = 6 #需要的聚类数
t0 = time.time()
kmeans_model = KMeans(n_clusters=types_num).fit(normalized_test_data)

plt.figure()
plt.xlabel('region')
plt.ylabel('written_bytes')
result = []
center = []
for i in range(types_num):
    temp = []
    result.append(temp)
for i, l in enumerate(kmeans_model.labels_):  # 画聚类点  i是下标，l是种类
    if i%1000 == 0:
        print('i: ',i)
    result[l].append(test_data[i].tolist())
    plt.scatter(data[i][0], data[i][4], c=colors[l], marker='o', s=20)

linelist = []
for i in range(types_num):
    center_0 =  list(list(kmeans_model.cluster_centers_)[i])[0] * std[0] + mean[0]
    center_1 =  list(list(kmeans_model.cluster_centers_)[i])[1] * std[1] + mean[1]
    center_2 = list(list(kmeans_model.cluster_centers_)[i])[2] * std[2] + mean[2]
    center_3 = list(list(kmeans_model.cluster_centers_)[i])[3] * std[3] + mean[3]
    #center_x = list(list(kmeans_model.cluster_centers_)[i])[0]
    #center_y = list(list(kmeans_model.cluster_centers_)[i])[1]
    print(len(result[i]),'|',center_0, '|',center_1, '|',center_2, '|',center_3)#聚类中心点
    #plt.scatter(center_x, center_y, c=colors[i], s=30)
    if (len(result[i]) > 10):#单个类点数量少于某个值则排除
        linelist.append([len(result[i]), center_0, center_1, center_3])
linelist.sort(key=(lambda x: x[3]))
linelist=linelist[-6:]
linelist.sort(key=(lambda x: x[2]))
linelist.sort(key=(lambda x: x[1]))
print(len(linelist))
print(linelist)


t1 = time.time()
print("时间:%.4fs" % (t1-t0))
plt.show()

