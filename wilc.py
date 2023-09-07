import os
import numpy
import scipy.stats as stats

path1 = '/home/lab/data/mdn/'
path2 = '/home/lab/data/mdnlinear/'
path3 = '/home/lab/data/mdnsum/'
alldata1 = []

alldata2 = []
for i in range(1,6):
    data1 = numpy.loadtxt(path1+'test'+str(i)+'.csv', dtype=float)
    data2 = numpy.loadtxt(path2+'test'+str(i)+'.csv', dtype=float)
    alldata1.append(data1[0])
    alldata2.append(data2[0])
print(alldata1,alldata2)
print(stats.wilcoxon(alldata1, alldata2, alternative='greater'))

