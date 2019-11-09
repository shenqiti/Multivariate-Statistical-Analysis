import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


INPUT_PATH = 'D:/python_code/NEW_step/pca.xls'
outputfile = 'D:/python_code/NEW_step/after_PCA.csv' #降维后的数据
data = pd.read_excel(INPUT_PATH)
print(data)


pca = PCA(3)
pca.fit(data)
com=pca.components_ #返回具有最大方差的成分。
pca=pca.explained_variance_ratio_ #返回各个成分各自的方差百分比
print (com,"百分比：\n",pca)
# print ("百分比：\n",pca)
# X = [i for i in range(1,7)]
# Y = pca
# plt.title("Percentage of variance for each component")
# plt.xlabel("Number of principal components")
# plt.ylabel("Percentage ")
# plt.plot(X,Y)
# for a, b in zip(X, Y):
#     plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom', fontsize=9)
# plt.show()

pca=PCA()
pca.fit(data)
low_d=pca.transform(data)#降维


pd.DataFrame(low_d).to_csv(outputfile)
pca.inverse_transform(low_d)

INPUT_PATH = 'D:/python_code/NEW_step/after_PCA.csv'
data = pd.read_csv(INPUT_PATH)
PC1 = data["0"]
PC2 = data["1"]


# PC1 = [-0.3733,-0.4819,-0.4398,0.4710,0.2980,0.3524]
# PC2 = [-0.2047,-0.2881,-0.5877,-0.4315,-0.3831,-0.4435]
# 
# 
# plt.scatter(PC1,PC2)    # X1 ,X2 ,X3,...
# plt.xlabel("PC1 (61.13%)")
# plt.ylabel("PC2 (21.19%)")
# i = 1
# for a, b in zip(PC1, PC2):
#     plt.text(a, b + 0.001, i, ha='center', va='bottom', fontsize=9)
#     i = i+1
# plt.show()

