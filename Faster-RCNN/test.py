import os

input_path = 'D:\datasets\PASCAL\VOCdevkit'
data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]

print(data_paths)