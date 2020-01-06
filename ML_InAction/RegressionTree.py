#********========回归树：Sklearn方法========********#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis = 0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))
# regr_1 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
# regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
# plt.figure()
# plt.scatter(X, y, c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
# rng = np.random.RandomState(1)
# X = np.linspace(0, 6, 100)[:, np.newaxis]
# y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
# regr_1 = DecisionTreeRegressor(max_depth = 4)
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 300, random_state = rng)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# y_1 = regr_1.predict(X)
# y_2 = regr_2.predict(X)
# plt.figure()
# plt.scatter(X, y, c = "k", label = "training samples")
# plt.plot(X, y_1, c = "g", label = "n_estimators=1", linewidth = 2)
# plt.plot(X, y_2, c = "r", label = "n_estimators=300", linewidth = 2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Boosted Decision Tree Regression")
# plt.legend()
# plt.show()

#********========CART：回归树案例========********#
def loadDataSet(filename):    # 函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数
	'''
	:param filename: 文件名
	:return:
		dataMat 每一行的数据集array类型
	'''
	numFeat = len(open(filename).readline().split('\t'))   # 所有特征的个数
	print('Number of File feature: ', numFeat)
	dataMat = []    # 假定最后一列是结果值
	fr = open(filename)
	for line in fr.readlines():    # 迭代每一行数据
		# lineArr = []
		# curLine = line.strip().split('\t')
		# for i in range(numFeat):
		# 	lineArr.append(float(curLine[i]))
		# dataMat.append(lineArr)
		lineArr = line.strip().split('\t')
		fltLine = [float(i) for i in lineArr]    #将每行转换成浮点数
		dataMat.append(fltLine)
	return dataMat

def binSplitDataSet(dataset, feature, value):    # 对数据集，按照feature列的value进行二元切分，dataset:二维array数组
	'''
	:param dataset: 数据集
	:param feature: 待切分的特征列
	:param value: 特征列需要比较的值
	:return:
		mat0 小于等于 value 的数据集在左边
		mat1 大于 value 的数据集在右边
	'''
	# m = dataset.shape[0]
	# left = []
	# right = []
	# for i in range(m):
	# 	if dataset[:, feature][i] < value:
	# 		left.append(dataset[:, feature][i])
	# 	else:
	# 		right.append(dataset[:, feature][i])
	# return left, right
	value_low = np.nonzero(dataset[:, feature] <= value)[0]   # np.nonzero(dataSet[:, feature] <= value)  返回结果为true行的index下标
	value_up = np.nonzero(dataset[:, feature] > value)[0]
	mat0 = dataset[value_low, :]
	mat1 = dataset[value_up, :]
	return mat0, mat1

def regLeaf(dataset):    # regLeaf 是产生叶节点的函数，就是求均值，即用聚类中心点来代表这类数据
	return np.mean(dataset[:, -1])    # 返回叶节点的均值

def regErr(dataset):    # 计算决策总方差，根据数据方差大小，划分数据到同一类中
	result = np.var(dataset[:, -1]) * np.shape(dataset)[0]
	return result

def chooseBestSplit(dataset, leafType=regLeaf, errType=regErr, ops=(1, 4)):    # 用最佳方式切分数据集，生成相应的叶节点
	'''
	:param dataset: 加载的原始数据集
	:param leafType: 建立叶节点的函数
	:param errType: 误差计算函数
	:param ops: 容许误差下降值，切分的最少样本数(ops决定了决策树划分停止的threshold值，被称为预剪枝；用于控制函数停止的时机)
	:return: bestIndex: feature的index坐标。bestValue: 切分的最优值
	'''
	tolS = ops[0]   # 当误差的下降值小于tolS时，决策树停止划分
	# ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
	# 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
	# 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
	tolN = ops[1]   # 当划分后的集合大小小于tolN时，决策树停止划分
	# dataSet[:, -1].T.tolist()[0] 取数据集的最后一列，转置为行向量，然后转换为list,取该list中的第一个元素
	if len(set(dataset[:, -1].T.tolist()[0])) == 1:    # 如果全部数据属于同一类别，则不用继续划分
		return None, leafType(dataset)
	m, n = np.shape(dataset)
	S = errType(dataset)    # 无分类误差的总方差和
	bestS, bestIndex, bestValue = np.inf, 0, 0
	for featIndex in range(n - 1):   # 循环每个特征的值
		# 下面的一行表示的是将某一列全部的数据转换为行，然后设置为list形式
		for splitVal in set(dataset[:, featIndex].T.tolist()[0]):    # 取某一列数据进行转置，并转化为列表
			mat0, mat1 = binSplitDataSet(dataset, featIndex, splitVal)    # 对该列进行分组，然后组内的成员的val值进行 二元切分
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):    # 判断二元切分后的数据量是否满足条件
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:   # 若切分算出的误差在可接受范围内，那么就记录切分点以及最小误差
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS:    # 判断二元切分的方式的元素误差是否符合预期
		return None, leafType(dataset)
	mat0, mat1 = binSplitDataSet(dataset, bestIndex, bestValue)
	# 对整体的成员进行判断，是否符合预期
	# 如果集合的 size 小于 tolN
	if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 当最佳划分后，集合过小，也不划分，产生叶节点
		return None, leafType(dataset)
	return bestIndex, bestValue

def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1, 4)): # 如果构建的是回归树，该模型是一个常数，如果是模型树，其模型师一个线性方程
	'''
	:param dataset: 加载的原始数据集
	:param leafType: 建立叶节点的函数
	:param errType: 计算误差函数
	:param ops: [容许误差下降值，切分的最少样本数]
	:return:
		retree: 决策树最后的结果
	'''
	feat, val = chooseBestSplit(dataset, leafType, errType, ops)   # 选择最优切分方式，feature索引值，最优切分值
	if feat is None:    # 如果spliting达到一个停止条件，则返回val
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	# 大于在右边，小于在左边，分为2个数据集
	lSet, rSet = binSplitDataSet(dataset, feat, val)
	# 递归的进行调用，在左右子树中继续递归生成树
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def isTree(obj):    # 判断节点是否为一个字典(现阶段树的结构以字典格式输出)
	return (type(obj).__name__ == 'dict')

def getMean(tree):   # 计算左右枝的均值
	'''
	:param tree: 输入的树
	:return: 返回tree节点的平均值
	'''
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

def prune(tree, testdata):    # 检查是否合适合并左右分枝
	'''
	:param tree: 待剪枝的树
	:param testdata: # 大于在右边，小于在左边，分为2个数据集
	:return:
		tree: 剪枝完成的树
	'''
	if np.shape(testdata)[0] == 0:    # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
		return getMean(tree)
	if (isTree(tree['right']) or isTree(tree['left'])):   # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
		lSet, rSet = binSplitDataSet(testdata, tree['spInd'], tree['spVal'])
	# 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	# 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)

	# 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点
	# 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
	# 1. 如果正确
	#   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
	#   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
	# 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testdata, tree['spInd'], tree['spVal'])
		# power(x, y)表示x的y次方
		errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(np.power(testdata[:, -1] - treeMean, 2))
		# 如果 合并的总方差 < 不合并的总方差，那么就进行合并
		if errorMerge < errorNoMerge:
			print("merging")
			return treeMean
		else:
			return tree
	else:
		return tree

def modelLeaf(dataset):    # 估计模型的系数
	'''
	:param dataset: 输入数据集
	:return: ws: 调用linearSolve函数，返回回归系数ws
	'''
	ws, X, Y = linearSolve(dataset)
	return ws

def modelErr(dataset):    # 求解模型的误差
	'''
	:param dataset: 输入数据集
	:return: 调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
	'''
	ws, X, Y = linearSolve(dataset)
	yHat = X * ws
	#print(np.corrcoef(yHat, Y, rowvar=0))
	return np.sum(np.power(Y - yHat, 2))

def linearSolve(dataset):    # 将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
	'''
	:param dataset: 输入数据集
	:return:
		ws: 线性回归的系数
		X: 格式化自变量X
		Y: 格式化目标变量Y
	'''
	m, n = np.shape(dataset)
	X = np.mat(np.ones((m, n)))    # 产生一个全为1的矩阵
	Y = np.mat(np.ones((m, 1)))
	X[:, 1: n] = dataset[:, 0: n - 1]    # X的0列为1，常数项，用于计算平衡误差
	Y = dataset[:, -1]
	xTx = X.T * X    # 转置矩阵*矩阵
	if np.linalg.det(xTx) == 0.0:    # 如果矩阵的逆不存在，会造成程序异常
		raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
	ws = xTx.I * (X.T * Y)    # 最小二乘法求最优解:  w0*1+w1*x1=y
	return ws, X, Y

def regTreeEval(model, indata):   # 对回归树进行预测， 为了和 modelTreeEval() 保持一致，保留两个输入参数
	'''
	:param model: 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
	:param indata: 输入的测试数据集
	:return:
		float(model): 将输入的模型数据转换为 浮点数 返回
	'''
	return float(model)    # 将输入模型的数据转化为浮点型

def modelTreeEval(model, indata):    #  对模型树进行预测
	# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
	# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
	'''
	:param model: 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型
	:param indata: 输入的测试数据
	:return:
		float(X * model) -- 将测试数据乘以回归系数得到一个预测值，转化为 浮点数 返回
	'''
	n = np.shape(indata)[1]
	X = np.mat(np.ones((1, n + 1)))
	X[:, 1: n + 1] = indata
	# print(X, model)
	return float(X * model)

def treeForeCast(tree, indata, modelEval=regTreeEval):    # 对特定模型的树进行预测，可以是 回归树 也可以是 模型树
	# 计算预测的结果
	# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
	# modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
	# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
	# 调用modelEval()函数，该函数的默认值为regTreeEval()
	'''
	:param tree: 已经训练好的树模型
	:param indata: 输入的测试数据集
	:param modelEval: 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
	:return: 返回预测值
	'''
	if not isTree(tree):
		return modelEval(tree, indata)
	if indata[tree['spInd']] <= tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], indata, modelEval)
		else:
			return modelEval(tree['left'], indata)
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'], indata, modelEval)
		else:
			return modelEval(tree['right'], indata)

def createForcast(tree, testData, modelEval=regTreeEval):     # 调用treeForeCast，对特定模型的树进行预测，可以是回归树也可以是模型树
	'''
	:param tree: 已经训练好的树的模型
	:param testData: 输入的测试数据集
	:param modelEval: 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
	:return: 返回测试数据集的预测值矩阵
	'''
	m = len(testData)
	yHat = np.mat(np.zeros((m, 1)))
	for i in range(m):
		yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
		print("yHat==>", yHat[i, 0])
	return yHat

if __name__ == '__main__':
	# # 测试数据集
	# testMat = np.mat(np.eye(4))
	# print(testMat)
	# print(type(testMat))
	# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
	# print(mat0, '\n-----------\n', mat1)

	# # 回归树
	# myDat = loadDataSet(r'D:\Data\ML_InAction\RegTree\data1.txt')
	# # myDat = loadDataSet(r'D:\Data\ML_InAction\RegTree\data2.txt')
	# # print 'myDat=', myDat
	# myMat = np.mat(myDat)
	# # print 'myMat=',  myMat
	# myTree = createTree(myMat)
	# print(myTree)
	#
	# # 1. 预剪枝就是：提起设置最大误差数和最少元素数
	# myDat = loadDataSet(r'D:\Data\ML_InAction\RegTree\data3.txt')
	# myMat = np.mat(myDat)
	# myTree = createTree(myMat, ops=(0, 1))
	# print(myTree)
	# #
	# # 2. 后剪枝就是：通过测试数据，对预测模型进行合并判断
	# myDatTest = loadDataSet(r'D:\Data\ML_InAction\RegTree\data3test.txt')
	# myMat2Test = np.mat(myDatTest)
	# #myTree = createTree(myMat2Test, ops=(0, 1))
	# myFinalTree = prune(myTree, myMat2Test)
	# print('\n\n-------------------')
	# print(myFinalTree)
	#
	# # --------
	# # 模型树求解
	# myDat = loadDataSet(r'D:\Data\ML_InAction\RegTree\data4.txt')
	# myMat = np.mat(myDat)
	# myTree = createTree(myMat, modelLeaf, modelErr)
	# print(myTree)
	#
	# 回归树 VS 模型树 VS 线性回归
	trainMat = np.mat(loadDataSet(r'D:\Data\ML_InAction\RegTree\bikeSpeedVslq_train.txt'))
	testMat = np.mat(loadDataSet(r'D:\Data\ML_InAction\RegTree\bikeSpeedVslq_test.txt'))
	# 回归树
	myTree1 = createTree(trainMat, ops=(1, 20))
	print(myTree1)
	yHat1 = createForcast(myTree1, testMat[:, 0])
	print("回归树:", np.corrcoef(yHat1, testMat[:, 1], rowvar=0)[0, 1])
	print("-------------------------"* 5)

	# 模型树
	myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
	yHat2 = createForcast(myTree2, testMat[:, 0], modelTreeEval)
	print(myTree2)
	print("模型树:", np.corrcoef(yHat2, testMat[:, 1], rowvar=0)[0, 1])
	print("-------------------------" * 5)

	# 线性回归
	ws, X, Y = linearSolve(trainMat)
	print(ws)
	m = len(testMat[:, 0])
	yHat3 = np.mat(np.zeros((m, 1)))
	for i in range(np.shape(testMat)[0]):
		yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
	print("线性回归:", np.corrcoef(yHat3, testMat[:, 1], rowvar=0)[0, 1])