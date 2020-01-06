#********========KNN: """机器学习实战：书中的例子（手写字识别、约会网站分类）"""========********#
import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from matplotlib.colors import ListedColormap
from collections import Counter
from sklearn import neighbors, datasets

def creatDataSet():
	"""创建数据集和标签"""
	group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataset, labels, k):
	# 方法1
	datasetSize = dataset.shape[0]   #  数据集有多少记录（即行数）
	diffMat = np.tile(inX, (datasetSize, 1)) - dataset   # tile()函数生成与dataset具有相同行的一个矩阵，具体查看tile()函数的应用方法
	sqDiffMat = diffMat ** 2   # 采用欧几里得距离计算
	sqDistance = sqDiffMat.sum(axis = 1)   #  求和
	distances = sqDistance ** 0.5   # np.sqrt(sqDistance)    开根号
	sortedDistIndicies = distances.argsort()     #   按照求得的距离排序
	# 开始计算排名前k个值的类别计数
	classCount = {}    # 使用字典的get(k, d)方法，相当于if...else...语句;参数k在字典中，返回list[k],参数不再字典中，返回d
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]    # 找到该样本的类型
		#classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
		if voteIlabel in classCount:
			classCount[voteIlabel] += 1
		else:
			classCount[voteIlabel] = 1
	# 按照统计数量的结果排序
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]   # 返回排序后的统计结果，排名最多的一类
	# # 实现方法2
	# # 计算距离
	# dist = np.sum((inX - dataset) ** 2, axis = 1) ** 0.5
	# k_labels = [labels[index] for index in dist.argsort()[0: k]]    # 对距离使用numpy中的argsort函数,函数返回的是索引，因此取前k个索引使用[0:k]
	# label = Counter(k_labels).most_common(1)[0][0]  # 使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple
	# return label

def test1():
	group, labels = creatDataSet()
	print(str(group))
	print(str(labels))
	print(classify0([0.1, 0.1], group, labels, 3))

def file2matrix(filename):
	numberofLines = len(open(filename).readlines())    # 获得文件中数据的行数
	returnMat = np.zeros((numberofLines, 3))    # 生成一个n行3列的空矩阵
	ClassLabelVecter = []    # 标签列表数据
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()   # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
		listFormLine = line.split('\t')       #  文件中的数据是以\t为分隔符的
		returnMat[index, :] = listFormLine[0:3]      #  将分割的前3列数据放入returnMat中，前3列为特征
		ClassLabelVecter.append(int(listFormLine[-1]))     #  每列的类别数据，就是 label 标签数据
		index += 1
	fr.close()
	return returnMat, ClassLabelVecter

def autoNorm(dataset):      #  归一化特征值，消除属性之间量级不同导致的影响
	# 求数据集中每列的最大值和最小值
	minvals = dataset.min(axis = 0)       #  axis = 0 表示按列求
	maxvals = dataset.max(axis = 0)
	ranges = maxvals - minvals    #  计算每一列的极差：最大值-最小值
	norm_dataset = (dataset - minvals) / ranges
	return norm_dataset, ranges, minvals

def datingClassTest():    #  对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
	hoRatio = 0.2    #  测试数据集的大小
	datingDataMat, datingLabel = file2matrix(r'D:\Data\ML_InAction\KNN\datingTestSet.txt')     # 从文件中加载数据集
	normMat, ranges, minVals = autoNorm(datingDataMat)     #  数据集归一化
	m = normMat.shape[0]     #  数据集行数
	numTestVecs = int(m * hoRatio)      # 测试数据集样本数
	print('numTestVecs = ', numTestVecs)
	errorCount = 0.0
	for i in range(numTestVecs):     #  对数据进行测试
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabel[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabel[i]))
		if (classifierResult != datingLabel[i]):
			errorCount += 1.0
	print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
	print(errorCount)

def img2vector(filename):     # KNN第二个项目案例的代码
	returnVector = np.zeros((1, 1024))     # 注：32 * 32 = 1024，对每一个文件进行转换
	fr = open(filename)
	for i in range(32):     # 一个文件中包含32行数据
		linestr = fr.readline()
		for j in range(32):     # 对每一行的每个记录进行处理
			returnVector[0, 32 * i + j] = int(linestr[j])     # 32*i+j表示特征数值在数组中的位置信息
	return returnVector     # 最终返回一个1*1024的一维矩阵

def handwritingClassTest():      # 对手写字识别的分类器
	hwLabels = []
	trainingFileList = os.listdir(r'data/2.KNN/trainingDigits')     # 导入目录路径下的所有文件
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):    # 对不同的文件进行处理
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector(r'data/2.KNN/trainingDigits/%s' % fileNameStr)

	# 导入测试数据集
	testFileList = os.listdir(r'data/2.KNN/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector(r'data/2.KNN/testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print("\nthe total number of errors is: %d" % errorCount)
	print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
	#test1()
	filename = r'D:\Data\ML_InAction\KNN\datingTestSet.txt'
	data, label = file2matrix(filename)
	datingClassTest()


#********=========KNN：sklearn算法========********#
import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 3
iris = datasets.load_iris()
X = iris.data[:, :2]    # 选取数据的前两列特征
y = iris.target
h = 0.02   # 网格步长
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])   # 创建彩色图
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	clf.fit(X, y)
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.show()