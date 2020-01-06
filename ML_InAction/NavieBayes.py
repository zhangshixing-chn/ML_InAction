"""贝叶斯分类器"""
# # 1、sklearn的贝叶斯模型
# import numpy as np
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
#
# X_Gau = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y_Gau = np.array([1, 1, 1, 2, 2, 2])
# X_Mul = np.random.randint(5, size=(6, 100))
# Y_Mul = np.array([1, 2, 3, 4, 5, 6])
# X_Ber = np.random.randint(2, size=(6, 100))
# Y_Ber = np.array([1, 2, 3, 4, 4, 5])
#
# clf_Gau = GaussianNB()
# clf_Mul = MultinomialNB()
# clf_Ber = BernoulliNB()
#
# clf_Gau.fit(X_Gau, Y_Gau)
# clf_Mul.fit(X_Mul, Y_Mul)
# clf_Ber.fit(X_Ber, Y_Ber)
#
# print(clf_Gau.predict([[-0.8, -1]]))
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X_Gau, Y_Gau, np.unique(Y_Gau))
# print(clf_pf.predict([[-0.8, -1]]))
# print(clf_Mul.predict(X_Mul[2:3]))
# print(clf_Ber.predict(X_Ber[2:3]))
import numpy as np
# *************===========Sample:屏蔽社区留言板的侮辱性言论===============****************
def load_dataSet():
	# postingList：单词列表
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	# classVec：标签列表，1表示侮辱性质，0表示非侮辱性质
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

def createVocabList(dataSet):   #  创建全部文档的词汇集
	# 获取所有单词的集合
	vocabSet = set([])
	for document in dataSet:    #  对数据集中的文档分解为集合
		vocabSet = vocabSet | set(document)    # 取两个集合的并集
	return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):   #  分析输入文档的词向量
	returnVec = [0] * len(vocabList)    #  创建一个和词汇集等长的向量，元素全部设为零
	for word in inputSet:    # 对输入文档的每个词汇进行判断
		if word in vocabList:    # 如果输入文档的词在词汇表中，则对应位置的值设为1
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec

def _trainNB0(trainMatrix, trainCategory):   #  训练模型1
	numTrainDocs = len(trainMatrix)   # 文件数
	numWords = len(trainMatrix[0])    # 单词数
	pAbusive = sum(trainCategory) / float(numTrainDocs)   # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
	# 构建单词出现的列表
	p0Num = np.zeros(numWords)  # [0,0,0,.....]
	p1Num = np.zeros(numWords)  # [0,0,0,.....]
	# 整个数据集单词出现的次数
	p0Denom = 0.0
	p1Denom = 0.0
	for i in range(numTrainDocs):    # 遍历所有的文件，如果是侮辱性文件，就计算此侮辱性文件中出现的侮辱性单词的个数
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num / p1Denom
	p0Vect = p0Num / p0Denom
	return p0Vect, p1Vect, pAbusive

def trainNB0(trainMatrix, trainCategory):   # 训练模型的优化版：优化出现概率为0、计算结果出现下溢问题
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	# 因为侮辱性的被标记为了1， 所以只要把他们相加就可以得到侮辱性的有多少
	# 侮辱性文件的出现概率，即train_category中所有的1的个数，
	# 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	# 单词出现的次数
	p0Num = np.ones(numWords)    # 改为np.ones()是为了防止数字过小溢出
	p1Num = np.ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
	p1Vect = np.log(p1Num / p1Denom)     #  去Log函数
	p0Vect = np.log(p0Num / p0Denom)
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	# 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
	# 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
	# 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
	# 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
	p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def bagOfWords2VecMN(vocabList, inputSet):
	returnVect = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVect[vocabList.index(word)] += 1    # 返回的是词向量出现的累计值
		else:
			print('the word: {} is not in my vocabulary'.format(word))
	return returnVect

def testingNB():
	listOPosts, listClasses = load_dataSet()   # 加载数据集
	myVocabList = createVocabList(listOPosts)  # 创建单词集合
	trainMat = []   # 计算单词是否出现，并创建数据矩阵
	for postinDoc in listOPosts:
		# 返回m*len(vocab_list)的矩阵， 记录的都是0，1信息
		# 其实就是那个东西的句子向量（就是data_set里面每一行,也不算句子吧)
		trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))    # 训练数据模型
	# 测试数据集
	testEntry1 = ['love', 'my', 'dalmation']
	thisDoc1 = np.array(setOfWord2Vec(myVocabList, testEntry1))
	print(testEntry1, 'classified as: ', classifyNB(thisDoc1, p0V, p1V, pAb))
	testEntry2 = ['stupid', 'garbage']
	thisDoc2 = np.array(setOfWord2Vec(myVocabList, testEntry2))
	print(testEntry2, 'classified as: ', classifyNB(thisDoc2, p0V, p1V, pAb))

# *************===========Sample:垃圾邮件过滤===============****************
def textParse(bigString):    #  切分文本
	import re
	listOfTokens = re.split(r'\W+', bigString)    # # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
	if len(listOfTokens) == 0:
		print(listOfTokens)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():    # 对贝叶斯垃圾邮件分类器进行自动化处理
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		# 切分解析数据，并归类为1
		# 添加垃圾邮件信息
		# 这里需要做一个说明，为什么我会使用try except 来做
		# 因为我们其中有几个文件的编码格式是 windows 1252　（spam: 17.txt, ham: 6.txt...)
		# 这里其实还可以 :
		# import os
		# 然后检查 os.system(' file {}.txt'.format(i))，看一下返回的是什么
		# 如果正常能读返回的都是：　ASCII text
		# 对于except需要处理的都是返回： Non-ISO extended-ASCII text, with very long lines
		try:
			words = textParse(open('data/4.NaiveBayes/email/spam/{}.txt'.format(i)).read())
		except:
			words = textParse(open('data/4.NaiveBayes/email/spam/{}.txt'.format(i), encoding='Windows 1252').read())
		docList.append(words)
		fullText.extend(words)
		classList.append(1)
		try:
			# 添加非垃圾邮件
			words = textParse(open('data/4.NaiveBayes/email/ham/{}.txt'.format(i)).read())
		except:
			words = textParse(open('data/4.NaiveBayes/email/ham/{}.txt'.format(i), encoding='Windows 1252').read())
		wordList = textParse(open(r'').read())
		docList.append(wordList)
		classList.append(1)
		# 切分解析数据，并归类为0
		wordList = textParse(open(r'').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)    # 创建词汇表
	trainingSet = range(50)
	testSet = []
	for i in range(10):
		randIndex = int(np.random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWord2Vec(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print('the errorCount is: ', errorCount)
	print('the testSet length is :', len(testSet))
	print('the error rate is :', float(errorCount) / len(testSet))

def testParseTest():
	print(textParse(open(r' ').read()))

# *************===========Sample:从个人广告中获取区域倾向===============****************
# 解析文本为词条向量
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def textParse1(bigString):   # 文本解析
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def calcMostFreq(vocabList,fullText):
	import operator
	freqDict={}
	for token in vocabList:  #遍历词汇表中的每个词
		freqDict[token]=fullText.count(token)  #统计每个词在文本中出现的次数
	sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)  #根据每个词出现的次数从高到底对字典进行排序
	return sortedFreq[:30]   #返回出现次数最高的30个单词

def localWords(feed1,feed0):
	import feedparser
	docList = []
	classList = []
	fullText = []
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])  # 每次访问一条RSS源
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList, fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList.remove(pairW[0])  # 去掉出现次数最高的那些词
	trainingSet = range(2 * minLen)
	testSet = []
	for i in range(20):
		randIndex = int(np.random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is:', float(errorCount) / len(testSet))
	return vocabList, p0V, p1V

def getTopWords(ny, sf):
	import operator
	vocabList, p0V, p1V = localWords(ny, sf)
	topNY = []
	topSF = []
	for i in range(len(p0V)):
		if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
		if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
	sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	for item in sortedSF:
		print(item[0])
	sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
	for item in sortedNY:
		print(item[0])

if __name__ == '__main__':
	testingNB()
	#spamTest()