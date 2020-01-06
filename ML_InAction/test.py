import os
import random
import pandas as pd
import numpy as np
# def image2Vec(filename):
# 	returnVec = np.zeros((1, 1024))
# 	fr = open(filename)
# 	for i in range(32):
# 		linstr = fr.readline()
# 		for j in range(32):
# 			returnVec[0, 32 * i + j] = int(linstr[j])   # 索引为0，因为returnVec是一个一维数组，下标是从0开始的
# 	return returnVec   #  得到的数组是一个二维数组
#
# if __name__ == '__main__':
# 	filename = r'D:\Data\ML_InAction\SVM\1.txt'
# 	dirname = r'D:\Data\ML_InAction\SVM'
# 	print(os.listdir(dirname))    # 路径名下的所有文件列表
# 	data = image2Vec(filename)
# 	print(data[0][:10])
# import tensorflow as tf
# print(tf.__version__)
random.seed(1)
data = np.random.randint(1, 10, 25).reshape(5, 5)
#print(data)
df_data = pd.DataFrame(data, columns = ['a', 'b', 'c', 'd', 'e'])
print(df_data)

def split_data(data, feature, value):
	m = data.shape[0]
	left = []
	right = []
	for i in range(m):
		if data[:, feature][i] < value:
			left.append(data[:, feature][i])
		else:
			right.append(data[:, feature][i])
	return left, right

def binSplitDataSet(dataset, feature, value):    # 对数据集，按照feature列的value进行二元切分
	value_low = np.nonzero(dataset[:, feature] <= value)[0]   # np.nonzero(dataSet[:, feature] <= value)  返回结果为true行的index下标
	value_up = np.nonzero(dataset[:, feature] > value)[0]
	mat0 = dataset[value_low, :]
	mat1 = dataset[value_up, :]
	return mat0, mat1

def split_df(dataset, feature, value):
	lower = dataset[dataset[feature] < value]
	up = dataset[dataset[feature] >= value]
	return lower, up

print(split_df(df_data, 'a', 5))