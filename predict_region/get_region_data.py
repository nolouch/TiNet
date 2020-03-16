import requests
import json
import time

url = 'http://10.233.57.252:2379/pd/api/v1/regions'
dir = 'region/'

while True:
    try:
        res = requests.get(url)#获取训练集
        data = json.loads(res.content.decode('utf-8'))
        filename = str(int(time.time())) + '.log'
        with open(dir + filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(filename)
        time.sleep(60)
    except:
        print('request url error')
        time.sleep(60)