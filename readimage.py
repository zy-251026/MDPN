import os
import torch
import pandas
import matplotlib.pyplot as plt
path = os.getcwd()
path2 = path + '/2023-06-26 18:03:431.csv'
path3 = path + '/2023-06-26 18:03:432.csv'
data1 = pandas.read_csv(path2, header=None)
data1 = torch.tensor(data1.values)
min = torch.min(data1)
max = torch.max(data1)
data1 = torch.div(data1-min,max-min)

data2 = pandas.read_csv(path3, header=None)
data2 = torch.tensor(data2.values)
min = torch.min(data2)
max = torch.max(data2)
data2 = torch.div(data2-min,max-min)
# data1[22,15] = 0.494
# data1[22,16] = 0.476
# data1[23,16] = 0.45
# data1[23,17] = 0.432
# data1[23,18] = 0.432
# data1[24,24] = 0.15
# data1[24,25] = 0.43
# data1[24,19] = 0.41
# data1[24,20] = 0.43
print(data2)
# data2[6,3] = 1
# data2[6,4] = 0.15
fig = plt.figure()
plt.subplot(1,2,1)
images = torch.stack((data1, data1, data1),2)
plt.imshow(images)
plt.subplot(1,2,2)
images = torch.stack((data2, data2, data2),2)
plt.imshow(images)
plt.show()

