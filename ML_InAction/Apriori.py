# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
#********========Apriori：算法实例========********#
import numpy as np

def loadDataSet():    # 加载数据集
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataset):   # 创建C1集合，即对原始数据去重、排序，转换为list，将元素变为frozenset（C1包含原始数据所有可能的取值）
	"""
	:param dataset: 原始数据集
	:return: frozenset 返回一个frozenset格式的list
	"""
	C1 = []
	for transaction in dataset:       # 遍历数据集的所有交易记录（每一行）
		for item in transaction:      # 遍历每一行元素项
			if not [item] in C1:      # 判断所有元素是否在C1中，如果不在，则添加到C1中
				C1.append([item])     # C1是一个集合，里面的元素也是一个集合，所以以[item]表示
	C1.sort()   # 从小到大的排序
	return list(map(frozenset, C1))   # frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用

def scanD(D, Ck, minSupport):
	"""
	:param D: 数据集
	:param Ck: 候选集列表
	:param minSupport: 最小支持度
	:return
	  retList: 支持度大于最小支持度的集合
	  supportData: 候选项集支持度
	"""
	ssCnt = {}   # 临时存放候选数据集Ck的频率，例如: a->10, b->5, c->8
	for tid in D:   # 遍历数据集的每一条交易记录
		for can in Ck:   # 遍历候选数据集的每一项
			# s.issubset(t)  测试是否 s 中的每一个元素都在 t 中
			if can.issubset(tid):   # 判断是否 can 中的每一个元素都在 tid 中; issubset()测试集合是否是另一个集合的子集
				if can not in ssCnt.keys():
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	#print('ssCnt: ', ssCnt)
	numItems = float(len(D))    # 数据集D的数量
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems   # 计算候选集的支持度
		if support >= minSupport:
			retList.insert(0, key)    # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
		supportData[key] = support
	return retList, supportData

def aprioriGen(Lk, k):   # 创建候选集Ck
	"""aprioriGen（输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
		例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
		仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作这是一个更高效的算法）
	Args:
		Lk 频繁项集列表
		k 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
	Returns:
		retList 元素两两合并的数据集
	"""
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):   # 利用两个循环，比较Lk中的元素之间构成的集合关系
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]   # [:k-2]  快速得到集合的并集结果，减少循环次数
			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])   # 取两个集合的并集
	#print(retList)
	return retList

def apriori(dataset, minSupport = 0.5):   # 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集
	"""
	:param dataset: 原始数据集
	:param minSupport: 最小支持度
	:return:
		L: 频繁项集全集
		supportData: 所有元素和支持度全集
	"""
	C1 = createC1(dataset)   # 创建数据集的C1频繁项集
	D = list(map(set, dataset))   # 数据集转化为集合,对每一行的记录去重
	L1, supportData = scanD(D, C1, minSupport)   # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
	#print('L1的值为：', L1)
	#print('supportData的取值为：', supportData)
	L = [L1]   # L 加了一层 list, L 一共 2 层 list
	k = 2
	#第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。
	# L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
	while (len(L[k-2]) > 0):   # 判断 L 的第 k-2 项的数据长度是否 > 0
		Ck = aprioriGen(L[k - 2], k)  # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
		#print('Ck的取值为：', Ck)
		Lk, supK = scanD(D, Ck, minSupport)  # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
		#print('Lk的取值为：', Lk)
		#print('supK的取值为：', supK)
		supportData.update(supK)  # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
		if len(Lk) == 0:
			break
		L.append(Lk) # Lk 表示满足频繁子项的集合，L 元素在增加，例如: l=[[set(1), set(2), set(3)]]; l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
		k += 1
	return L, supportData

def generateRules(L, supportData, minConf=0.7):   # 生成关联规则
	"""generateRules
	    Args:
	        L 频繁项集列表
	        supportData 频繁项集支持度的字典
	        minConf 最小置信度
	    Returns:
	        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
	    """
	bigRuleList = []
	# 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])],
	#       [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
	for i in range(1, len(L)):
		for freqSet in L[i]:   # 获取频繁项集中每个组合的所有元素
			# 假设:freqSet=frozenset([1,3]),H1=[frozenset([1]),frozenset([3])];组合总的元素并遍历子元素，并转化为frozenset集合，再存放到list列表中
			H1 = [frozenset([item]) for item in freqSet]
			# 2 个的组合，走 else, 2 个以上的组合，走 if
			if (i > 1):
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	"""calcConf（对两个元素的频繁项，计算可信度，例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件）
	Args:
		freqSet 频繁项集中的元素，例如: frozenset([1, 3])
		H 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
		supportData 所有元素的支持度的字典
		brl 关联规则列表的空数组
		minConf 最小可信度
	Returns:
		prunedH 记录 可信度大于阈值的集合
	"""
	prunedH = []
	for conseq in H:
		"""
		支持度定义: a -> b = support(a | b) / support(a). 
		假设 freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，那么 frozenset([1]) 至 frozenset([3]) 的可信度为
		support(a | b) / support(a) = supportData[freqSet]/supportData[freqSet-conseq] = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
		"""
		conf = supportData[freqSet] / supportData[freqSet - conseq]
		if conf >= minConf:
			# 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
			print(freqSet - conseq, '-->', conseq, 'conf:', conf)
			brl.append((freqSet - conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	"""rulesFromConseq
		Args:
			freqSet 频繁项集中的元素，例如: frozenset([2, 3, 5])
			H 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
			supportData 所有元素的支持度的字典
			brl 关联规则列表的数组
			minConf 最小可信度
		"""
	# H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
	# 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
	# 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
	# 那么 m = len(H[0]) 的递归的值依次为 1 2
	# 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，没必要再计算 freqSet 与 H[0] 的关联规则了。
	m = len(H[0])
	if (len(freqSet) > (m + 1)):
		# print 'freqSet******************', len(freqSet), m + 1, freqSet, H, H[0]
		# 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
		# 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
		# 第二次 。。。没有第二次，递归条件判断时已经退出了
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 返回置信度大于最小可信度的集合
		#print('Hmp1=', Hmp1)
		#print('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
		# 计算置信度后，还有数据大于最小置信度的话，那么继续递归调用，否则跳出递归
		if (len(Hmp1) > 1):
			# print('----------------------', Hmp1)
			# print(len(freqSet), len(Hmp1[0]) + 1)
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def testApriori():
	dataset = loadDataSet()
	print('DataSet:', dataset)
	# Apriori 算法生成频繁项集以及它们的支持度
	L1, supportData1 = apriori(dataset, minSupport = 0.7)
	print('L(0.7): ', L1)
	print('supportData(0.7): ', supportData1)
	print('-*-' * 50)
	# Apriori 算法生成频繁项集以及它们的支持度
	L2, supportData2 = apriori(dataset, minSupport = 0.5)
	print('L(0.5): ', L2)
	print('supportData(0.5): ', supportData2)

def testGenerateRules():
	# 加载测试数据集
	dataSet = loadDataSet()
	#print ('dataSet: ', dataSet)
	# Apriori 算法生成频繁项集以及它们的支持度
	L1, supportData1 = apriori(dataSet, minSupport=0.5)
	print ('L(0.5): ', L1)
	#print ('supportData(0.5): ', supportData1)
	# 生成关联规则
	rules = generateRules(L1, supportData1, minConf=0.5)
	print ('rules: ', rules)

if __name__ == '__main__':
	#testApriori()
	#testGenerateRules()
	filename = r'D:\Data\ML_InAction\Apriori\mushroom.txt'
	dataset = [line.split() for line in open(filename).readlines()]
	L, supportData = apriori(dataset, minSupport=0.3)
	for item in L[1]:
		if item.intersection('2'):
			print(item)
	for item in L[2]:
		if item.intersection('2'):
			print(item)
	#rules = generateRules(L, supportData, minConf=0.8)
	#print('rules: ', rules)