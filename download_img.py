import os
from urllib.request import urlretrieve

file = open('imagenet_tiger.txt')

for i, l in enumerate(file.readlines()):
    url = l[:-1]
    print(i)
    try:
        urlretrieve(url, 'E:\\Mulong\\Datasets\\Trail_NN\\imgnet\\tiger\\' + str(i) + '.png')
    except:
        print('bad url')
