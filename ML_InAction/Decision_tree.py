#********========Decision Tree: 机器学习实战：决策树算法========********#
import operator
from math import log
import ML_InAction.descionTreePlot as dtPlot
from collections import Counter

def creatDataSet():
	dataSet = [[1, 1, 'yes'],
	           [1, 1, 'yes'],
	           [1, 0, 'no'],
	           [0, 1, 'no'],
	           [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']     # labels的两种取值
	return dataSet, labels

def calcShannonEnt(dataSet):
	# # 计算香农熵的方式1：
	# numEntries = len(dataSet)   # 求list的长度，表示计算参与训练的数据量
	# labelCounts = {}    # 创建字典，记录标签label出现的次数
	# for featVec in dataSet:
	# 	currentLabel = featVec[-1]     # 当前记录的标签存储，即每一行数据最后一个数据代表标签
	# 	if currentLabel not in labelCounts.keys():    # 为当前键创建字典对象，并统计当前类别出现的次数
	# 		labelCounts[currentLabel] = 0
	# 	else:
	# 		labelCounts[currentLabel] += 1
	# shannonEnt = 0.0   # 对于label标签的占比，求出Label标签的香农熵
	# for key in labelCounts:
	# 	prob = float(labelCounts[key]) / float(numEntries)    #  计算不同类别出现的概率，通过频率计算
	# 	shannonEnt -= prob * log(prob, 2)   # 计算香农熵，取对数
	# 计算香农熵的方式2：
	label_count = Counter(data[-1] for data in dataSet)     #  统计标签出现的次数
	probs = [p[1] / len(dataSet) for p in label_count.items()]     #  计算概率
	shannonEnt = sum([-p * log(p, 2) for p in probs])     # 计算香农熵
	return shannonEnt

def splitDataSet(dataSet, index, value):
	# 方式1：
	retDataSet = []
	for featVec in dataSet:   # index列为value的数据集
		if featVec[index] == value:    # 判断index列的值是否等于value
			reducedFeatVec = featVec[:index]    # 选取featVec的前index行
			reducedFeatVec.extend(featVec[index+1:])    # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
			retDataSet.append(reducedFeatVec)   # append将添加内容看做一个对象，extend将添加内容看做一个序列；两者存在一定的区别
	# 方式2
	# retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
	return retDataSet

def chooseBestFeatureToSplit(dataSet):     # 选择最好的特征作为分类变量
	# 方式1
	numfeatures = len(dataSet[0]) - 1    # 求第一行有多少列的 Feature, 最后一列是label列
	baseEntropy = calcShannonEnt(dataSet)     # label的信息熵
	bestInfoGain = 0.0   # 最优的信息增益值,
	bestFeature = -1     # 最优的Featurn编号
	# 迭代所有的特征
	for i in range(numfeatures):
		featlist = [example[i] for example in dataSet]     # 获取每个实例的第i+1个feature,组成list
		uniqueVals = set(featlist)     # 获取去重后的集合
		newEntropy = 0.0    # 创建一个临时信息熵
		for value in uniqueVals:     # 遍历某一列的value集合，计算该列的信息熵。遍历当前特征中的所有唯一属性值，
			# 对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		# gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
		# 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
		infoGain = baseEntropy - newEntropy     #  计算信息增益
		print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
	# # 计算初始香农熵
	# base_entropy = calcShannonEnt(dataSet)
	# best_info_gain = 0
	# best_feature = -1
	# # 遍历每一个特征
	# for i in range(len(dataSet[0]) - 1):
	#     # 对当前特征进行统计
	#     feature_count = Counter([data[i] for data in dataSet])
	#     # 计算分割后的香农熵
	#     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
	#                    for feature in feature_count.items())
	#     # 更新值
	#     info_gain = base_entropy - new_entropy
	#     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
	#     if info_gain > best_info_gain:
	#         best_info_gain = info_gain
	#         best_feature = i
	# return best_feature

def majorityCnt(classList):     #  选择出现次数最多的一个结果
	# 方法1
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		else:
			classCount[vote] += 1
	#  倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]
	# # 方法2：
	# major_label = Counter(classList).most_common(1)[0]
	# return major_label

def createTree(dataSet, labels):    # 创建决策树
	classlist = [example[-1] for example in dataSet]
	# 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
	# 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
	# count() 函数是统计括号中的值在list中出现的次数
	if classlist.count(classlist[0]) == len(classlist):
		return classlist[0]
	# 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
	# 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
	if len(dataSet[0]) == 1:
		return majorityCnt(classlist)
	bestFeat = chooseBestFeatureToSplit(dataSet)    # 选择最优的列，得到最优列对应的label含义
	bestFeatLabel = labels[bestFeat]   # 获取label的名称
	myTree = {bestFeatLabel:{}}   #  初始化Tree结构
	# 注：labels列表是可变对象，在python函数中作为参数时传址引用，能够被全局修改
	# 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
	del (labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]    # 取出最优列，然后它的branch做分类
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]     # 求出剩余的标签label
		# 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree

def classify(inputTree, featLabels, testVec):    #  对新数据进行分类
	firstStr = list(inputTree.keys())[0]  # 获取根节点对应的key值
	secondDict = inputTree[firstStr]      # 通过key得到根节点对应的value
	# 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
	featIndex = featLabels.index(firstStr)
	# 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
	key = testVec[featIndex]
	valueOfFeat = secondDict[key]
	print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
	# 判断分枝是否结束: 判断valueOfFeat是否是dict类型
	if isinstance(valueOfFeat, dict):
		classLabel = classify(valueOfFeat, featLabels, testVec)
	else:
		classLabel = valueOfFeat
	return classLabel

def storeTree(inputTree, filename):    # 保存树结构模型
	import pickle
	# 第一种方法
	fw = open(filename, 'wb')    #  打开需要写入的文件位置
	pickle.dump(inputTree, fw)      # 写入模型
	fw.close()
	# 第二种方法
	# with open(filename, 'wb') as fw:
	# 	pickle.dump(inputTree, fw)

def grabTree(filename):    # 将之前存储的决策树模型使用 pickle 模块 还原出来
	import pickle
	fr = open(filename,'rb')
	return pickle.load(fr)

def fishTest():
	myDat, labels = creatDataSet()
	calcShannonEnt(myDat)
	print(chooseBestFeatureToSplit(myDat))
	import copy
	myTree = createTree(myDat, copy.deepcopy(labels))
	print(myTree)
	print(classify(myTree, labels, [1, 1]))
	print(get_tree_height(myTree))
	dtPlot.createPlot(myTree)

def ContactLensesTest():
	fr = open(r'D:\Data\ML_InAction\DescionTree\lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = createTree(lenses, lensesLabels)
	print(lensesTree)
	dtPlot.createPlot(lensesTree)

def get_tree_height(tree):
	if not isinstance(tree, dict):
		return 1
	child_trees = list(tree.values())[0].values()
	# 遍历子树, 获得子树的最大高度
	max_height = 0
	for child_tree in child_trees:
		child_tree_height = get_tree_height(child_tree)
		if child_tree_height > max_height:
			max_height = child_tree_height
	return max_height + 1

if __name__ == '__main__':
	#fishTest()
	ContactLensesTest()