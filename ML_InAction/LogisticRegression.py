import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#********==========Logistic回归在简单数据集上的分类==========********#
# 导入数据集
def loadDataSet(filename):
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		linArr = line.strip().split()
		if len(linArr) == 1:
			continue     # 如果只有一个空元素（没有值），则跳过此循环
		dataMat.append([1.0, float(linArr[0]), float(linArr[1])])   # 添加1.0代表逻辑回归的常数项
		labelMat.append(int(linArr[2]))
	fr.close()
	return dataMat, labelMat

# sigmoid跃阶函数
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))
	#return 2 * 1.0 / (1 + np.exp(-2 * inX)) - 1   # tanh函数

def gradAscent(dataMatIn, classLabels):   #  正常计算的梯度上升法
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()   # 首先将数组转为矩阵，并将行向量转化为列向量
	m, n = dataMatrix.shape[0], dataMatrix.shape[1]
	alpha = 0.01  # 变化步长
	maxCycles = 700    # 迭代次数控制
	weights = np.ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)    # h表示的是一个列向量，为n*1, 进行矩阵的乘法计算
		error = (labelMat - h)        #  向量相减
		weights = weights + alpha * dataMatrix.transpose() * error     #  矩阵乘法，返回最终的回归系数
	return np.array(weights)

def stocGradAscent0(dataMatrix, classLabels):    # 随机梯度上升法：只能使用一个样本点更新回归系数
	m, n = np.shape(dataMatrix)
	alpha = 0.1
	weights = np.ones(n)    #  初始化长度为N的数组，其元素全部为1
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))    #  h 是一个具体的数值，而不是一个矩阵
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]    #  梯度上升法的权重更新方式
	return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 100):      # 随机梯度上升法，采用随机样本更新回归系数
	m, n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0+j+i) + 0.0001    # i和j的不断增大，导致alpha的值不断减少，但是不为0
			# 随机产生一个 0～len()之间的一个值
			# random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
			randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机产生一个值，进行回归系数的更新迭代
			h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
			error = classLabels[dataIndex[randIndex]] - h
			weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
			del (dataIndex[randIndex])
	return weights

def plotBestFit(dataArr, labelMat, weights):    # 分类结果的可视化
	n = dataArr.shape[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:    #  将对应的分类结果保存在列表中
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')    #  第一幅散点图
	ax.scatter(xcord2, ycord2, s=30, c='green')    #  第二幅散点图
	x = np.arange(-3.0, 3.0, 0.1)
	# dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]); w0 * x0 + w1 * x1 + w2 * x2 = f(x)
	# x0 最开始就设置为1，x2 就是我们画图的y值，而f(x) 被我们磨合误差给算到w0, w1, w2身上去了
	# 所以：w0 + w1 * x + w2 * y = 0 = > y = (-w0 - w1 * x) / w2
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)     #  直线图
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

def sampleTest():
	dataMat, labelMat = loadDataSet(r'D:\Data\ML_InAction\LogisticRegression\TestSet.txt')
	dataArr = np.array(dataMat)
	#weights = stocGradAscent0(dataArr, labelMat)
	#weights = stocGradAscent1(dataArr, labelMat)
	weights = gradAscent(dataArr, labelMat)
	plotBestFit(dataArr, labelMat, weights)

#*******========Logistic回归在马疾病的分类==========********
def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():   # 对数据集进行格式化处理
	frTrain = open(r'D:\Data\ML_InAction\LogisticRegression\HorseColicTraining.txt')
	frTest = open(r'D:\Data\ML_InAction\LogisticRegression\HorseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
	errorCount = 0
	numTestVec = 0.0
	# 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount) / numTestVec)
	print("the error rate of this test is: %f" % errorRate)
	return errorRate

def multitest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

if __name__ == '__main__':
	filename = r'D:\Data\ML_InAction\LogisticRegression\TestSet.txt'
	sampleTest()
	#multitest()