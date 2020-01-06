#********========SVM：sklearn中的算法========********#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# def load_dataset(filename):    # 载入数据集
# 	dataMat = []    #  特征数据
# 	labelMat = []    #  标签数据
# 	fr = open(filename)
# 	for line in fr.readlines():
# 		linArr = line.strip().split('\t')
# 		dataMat.append([float(linArr[0]), float(linArr[1])])
# 		labelMat.append(float(linArr[2]))
# 	return dataMat, labelMat
# # X, Y = load_dataset(r'D:\Data\ML_InAction\SVM\dataSet.txt')
# # X = np.mat(X)
# # clf = SVC(kernel = 'linear')
# # clf.fit(X, Y)
# # w = clf.coef_[0]  # 获取分割超平面:[ 0.81444269 -0.27274371]
# # a = -w[0] / w[1]  # 其中a等于0.81444269/-0.27274371
# # xx = np.linspace(-2, 10)
# # yy = a * xx - (clf.intercept_[0]) / w[1]
# # # 通过支持向量绘制分割超平面
# # print("support_vectors_ = ", clf.support_vectors_)
# # b = clf.support_vectors_[0]
# # yy_down = a * xx + (b[1] - a * b[0])
# # b = clf.support_vectors_[-1]
# # yy_up = a * xx + (b[1] - a * b[0])
# # plt.plot(xx, yy, 'k-')
# # plt.plot(xx, yy_down, 'k--')
# # plt.plot(xx, yy_up, 'k--')
# # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolors = 'none')
# # plt.scatter(X[:, 0].flat, X[:, 1].flat, c = Y, cmap = plt.cm.Paired)   # flat返回的是一个迭代器，可以用for访问数组每一个元素
# # plt.axis('tight')
# # plt.show()
#
# #*********========SVM：Simple版本========********#
# def selectJrand(i, m):    # 随机选择一个整数, i表示第一个alpha的下标，m表示所有alpha的数目,返回一个不为i的随机数，在0~m之间
# 	j = i
# 	while j == i:
# 		j = int(np.random.uniform(0, m))    #  只要函数值不为i，怎函数会随机选择一个j
# 	return j
#
# def clipAlpha(aj, H, L):  # 调整aj的值，使之处于L 和 H 之间
# 	if aj > H:
# 		aj = H    #  当aj过大，则把最大值 H 赋给它
# 	if L > aj:
# 		aj = L    #  当aj过小，则把最小值 L 赋给它
# 	return aj
#
# def smpSimple(dataMatIn, classLabels, C, toler, maxIter):
# 	"""smoSimple
# 	    Args:
# 	        dataMatIn    数据集
# 	        classLabels  类别标签
# 	        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
# 	            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
# 	            可以通过调节该参数达到不同的结果。
# 	        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
# 	        maxIter 退出前最大的循环次数
# 	    Returns:
# 	        b       模型的常量值
# 	        alphas  拉格朗日乘子
# 	"""
# 	dataMatrix = np.mat(dataMatIn)
# 	labelMat = np.mat(classLabels).transpose()  # 矩阵转置，效果等同于.T
# 	m, n = np.shape(dataMatrix)
# 	b = 0     # 初始化 b和alphas(alpha有点类似权重值)
# 	alphas = np.mat(np.zeros((m, 1)))
# 	iter = 0  # 没有任何alpha改变情况下遍历数据的次数
# 	while (iter < maxIter):
# 		alphaPairsChanged = 0     # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
# 		for i in range(m):
# 			# 预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
# 			fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
# 			Ei = fXi - float(labelMat[i])     # 预测结果与真实结果比对，计算误差Ei
# 			if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):   # 是否满足KKT条件
# 				j = selectJrand(i, m)    # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
# 				fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b   # 预测j的结果
# 				Ej = fXj - float(labelMat[j])
# 				alphaIold = alphas[i].copy()
# 				alphaJold = alphas[j].copy()
# 				# L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
# 				# labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
# 				if (labelMat[i] != labelMat[j]):
# 					L = max(0, alphas[j] - alphas[i])
# 					H = min(C, C + alphas[j] - alphas[i])
# 				else:
# 					L = max(0, alphas[j] + alphas[i] - C)
# 					H = min(C, alphas[j] + alphas[i])
# 				# 如果相同，就没发优化了
# 				if L == H:
# 					print("L==H")
# 					continue
# 				# eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
# 				eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
# 				if eta >= 0:
# 					print("eta>=0")
# 					continue
# 				alphas[j] -= labelMat[j] * (Ei - Ej) / eta     # 计算出一个新的alphas[j]值
# 				alphas[j] = clipAlpha(alphas[j], H, L)     # 并使用辅助函数，以及L和H对其进行调整
# 				if (abs(alphas[j] - alphaJold) < 0.00001):     # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环
# 					print("j not moving enough")
# 					continue
# 				alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])    # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
# 				# 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
# 				# w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
# 				# 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
# 				# 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
# 				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
# 					j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
# 				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
# 					j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
# 				if (0 < alphas[i]) and (C > alphas[i]):
# 					b = b1
# 				elif (0 < alphas[j]) and (C > alphas[j]):
# 					b = b2
# 				else:
# 					b = (b1 + b2) / 2.0
# 				alphaPairsChanged += 1
# 				print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
# 		# 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序   # 知道更新完毕后，iter次循环无变化，才推出循环。
# 		if (alphaPairsChanged == 0):
# 			iter += 1
# 		else:
# 			iter = 0
# 		print("iteration number: %d" % iter)
# 	return b, alphas
#
# def calcWs(alphas, dataArr, classLabels):    # 基于alpha的值，计算w的值,
# 	X = np.mat(dataArr)
# 	labelMat = np.mat(classLabels).transpose()
# 	m, n = X.shape
# 	w = np.zeros((n,1))
# 	for i in range(m):
# 		w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
# 	return w
#
# def plotfig_SVM(xMat, yMat, ws, b, alphas):
# 	xMat = np.mat(xMat)
# 	yMat = np.mat(yMat)
# 	b = np.array(b)[0]   # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])   #  flatten()函数的用法
# 	x = np.arange(-1.0, 10.0, 0.1)
# 	y = (-b - ws[0, 0] * x) / ws[1, 0]
# 	ax.plot(x, y)
# 	for i in range(np.shape(yMat[0,:])[1]):
# 		if yMat[0, i] > 0:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
# 		else:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
# 	for i in range(100):
# 		if alphas[i] > 0.0:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
# 	plt.show()
#
# if __name__ == '__main__':
# 	# 获取特征和目标变量
# 	dataArr, labelArr = load_dataset(r'D:\Data\ML_InAction\SVM\dataSet.txt')
# 	# print(labelArr)
#
# 	# b是常量值， alphas是拉格朗日乘子
# 	b, alphas = smpSimple(dataArr, labelArr, 0.6, 0.001, 40)
# 	print('/n/n/n')
# 	print('b=', b)
# 	print('alphas[alphas>0]=', alphas[alphas > 0])
# 	print('shape(alphas[alphas > 0])=', np.shape(alphas[alphas > 0]))
# 	for i in range(100):
# 		if alphas[i] > 0:
# 			print(dataArr[i], labelArr[i])
# 	# 画图
# 	ws = calcWs(alphas, dataArr, labelArr)
# 	plotfig_SVM(dataArr, labelArr, ws, b, alphas)


# #*********========SVM：Non_Kernel版本========********#
# class optStruct:
# 	def __init__(self, dataMatIn, classLabels, C, toler):
# 		self.X = dataMatIn
# 		self.labelMat = classLabels
# 		self.C = C
# 		self.tol = toler
# 		self.m = np.shape(dataMatIn)[0]
# 		self.alphas = np.mat(np.zeros((self.m, 1)))
# 		self.b = 0
# 		self.eCache = np.mat(np.zeros((self.m, 2)))    # eCache的第一列为是否有效的标志位，第二列为实际的E值
#
# def load_dataSet(fileName):
# 	'''
# 	loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）
# 	    Args:
# 	        fileName 文件名
# 	    Returns:
# 	        dataMat  数据矩阵
# 	        labelMat 类标签
# 	'''
# 	dataMat = []
# 	labelMat = []
# 	fr = open(fileName)
# 	for line in fr.readlines():
# 		lineArr = line.strip().split('\t')
# 		dataMat.append([float(lineArr[0]), float(lineArr[1])])
# 		labelMat.append(float(lineArr[2]))
# 	return dataMat, labelMat
#
# def selectJrand(i, m):
# 	'''
# 	 随机选择一个整数
# 	Args:
# 	    i  第一个alpha的下标
# 	    m  所有alpha的数目
# 	Returns:
# 		j   返回一个不为i的随机数，在0~m之间的整数值
# 	'''
# 	j = i
# 	while j == i:
# 		j = np.random.randint(0, m - 1)
# 	return j
#
# def clipAlpha(aj, H, L):
# 	'''
# 	clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
# 	    Args:
# 	        aj  目标值
# 	        H   最大值
# 	        L   最小值
# 	    Returns:
# 	        aj  目标值
# 	'''
# 	aj = min(aj, H)
# 	aj = max(L, aj)
# 	return aj
#
# def calcEk(oS, k):   # 求Ek的误差，真实值与预测值的差值
# 	'''
# 	Args:
# 	    oS  optStruct对象
# 	    k   具体的某一行
# 	Returns:
# 	    Ek  预测结果与真实结果比对，计算误差Ek
# 	'''
# 	fXk = np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k].T) + oS.b
# 	Ek = fXk - float(oS.labelMat[k])
# 	return Ek
#
# def selectJ(i, oS, Ei):
# 	maxK = -1
# 	maxDeltaE = 0
# 	Ej = 0
# 	oS.eCache[i] = [1, Ei]
# 	validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
# 	if (len(validEcacheList)) > 1:
# 		for k in validEcacheList:
# 			if k == i:
# 				continue
# 			Ek = calcEk(oS, k)
# 			deltaE = abs(Ei - Ek)
# 			if deltaE > maxDeltaE:
# 				maxK = k
# 				maxDeltaE = deltaE
# 				Ej = Ek
# 		return maxK, Ej
# 	else:
# 		j = selectJrand(i, oS.m)
# 		Ej = calcEk(oS, j)
# 	return j, Ej
#
# def updateEk(oS, k):
# 	Ek = calcEk(oS, k)
# 	oS.eCache[k] = [1, Ek]
#
# def innerL(i, oS):
# 	Ei = calcEk(oS, i)
# 	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
# 		j, Ej = selectJ(i, oS, Ei)
# 		alphaIold = oS.alphas[i].copy()
# 		alphaJold = oS.alphas[j].copy()
# 		if oS.labelMat[i] != oS.labelMat[j]:
# 			L = max(0, oS.alphas[j] - oS.alphas[i])
# 			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
# 		else:
# 			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
# 			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
# 		if L == H:
# 			print("L==H")
# 			return 0
# 		eta = oS.X[i] - oS.X[j]
# 		eta = - eta * eta.T
# 		if eta >= 0:
# 			print("eta>=0")
# 			return 0
# 		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
# 		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
# 		updateEk(oS, j)
# 		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
# 			print("j not moving enough")
# 			return 0
# 		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
# 		updateEk(oS, i)
# 		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[i].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i] * oS.X[j].T
# 		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[j].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j] * oS.X[j].T
# 		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
# 			oS.b = b1
# 		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
# 			oS.b = b2
# 		else:
# 			oS.b = (b1 + b2) / 2
# 		return 1
# 	else:
# 		return 0
#
#
# def smoP(dataMatIn, classLabels, C, toler, maxIter):
# 	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
# 	iter = 0
# 	entireSet = True
# 	alphaPairsChanged = 0
# 	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
# 		alphaPairsChanged = 0
# 		if entireSet:
# 			for i in range(oS.m):
# 				alphaPairsChanged += innerL(i, oS)
# 				print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
# 			iter += 1
# 		else:
# 			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
# 			for i in nonBoundIs:
# 				alphaPairsChanged += innerL(i, oS)
# 				print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
# 			iter += 1
# 		if entireSet:
# 			entireSet = False
# 		elif alphaPairsChanged == 0:
# 			entireSet = True
# 		print("iteration number: %d" % iter)
# 	return oS.b, oS.alphas
#
# def calcWs(alphas, dataArr, classLabels):
# 	X = np.mat(dataArr)
# 	labelMat = np.mat(classLabels).T
# 	m, n = np.shape(X)
# 	w = np.zeros((n, 1))
# 	for i in range(m):
# 		w += np.multiply(alphas[i] * labelMat[i], X[i].T)
# 	return w
#
# def plotfig_SVM(xArr, yArr, ws, b, alphas):
# 	xMat = np.mat(xArr)
# 	yMat = np.mat(yArr)
# 	b = np.array(b)[0]
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
# 	x = np.arange(-1.0, 10.0, 0.1)
# 	y = (- b - ws[0, 0] * x) / ws[1, 0]
# 	ax.plot(x, y)
# 	for i in range(np.shape(yMat[0])[1]):
# 		if yMat[0, i] > 0:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
# 		else:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
# 	for i in range(100):
# 		if alphas[i] > 0.0:
# 			ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
# 	plt.show()
#
# if __name__ == "__main__":
# 	dataArr, labelArr = load_dataSet(r'D:\Data\ML_InAction\SVM\dataSet.txt')
# 	b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
# 	print('/n/n/n')
# 	print('b=', b)
# 	print('alphas[alphas>0]=', alphas[alphas > 0])
# 	print('shape(alphas[alphas > 0])=', np.shape(alphas[alphas > 0]))
# 	for i in range(100):
# 		if alphas[i] > 0:
# 			print(dataArr[i], labelArr[i])
# 	ws = calcWs(alphas, dataArr, labelArr)
# 	plotfig_SVM(dataArr, labelArr, ws, b, alphas)

#********========SVM：complete版本========********#
class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		'''
		dataMatIn    数据集
		classLabels  类别标签
		C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
			可以通过调节该参数达到不同的结果。
		toler   容错率
		kTup    包含核函数信息的元组, 如('rbf', 10)
		'''
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = np.shape(dataMatIn)[0]      # 数据的行数
		self.alphas = np.mat(np.zeros((self.m, 1)))      # 待估计的特征回归系数
		self.b = 0      # 待估计的常数项
		self.eCache = np.mat(np.zeros((self.m, 2)))     # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
		self.K = np.mat(np.zeros((self.m, self.m)))     # m行m列的矩阵
		for i in range(self.m):
			self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)

def kernelTrans(X, A, kTup):    #  核转换函数，将数据由低维转化成高维度空间数据
	'''
	:param X: dataMatIn数据集
	:param A: dataMatIn数据集的第i行
	:param kTup: 和函数信息，是一个元祖
	'''
	m, n = np.shape(X)
	K = np.mat(np.zeros((m, 1)))
	if kTup[0] == 'lin':     #  线性核函数
		K = X * A.T
	elif kTup[0] == 'rbf':    #  径向基高斯核函数
		for j in range(m):
			deltaRow = X[j, :] - A
			K[j] = deltaRow * deltaRow.T
		K = np.exp(K / (-1 * kTup[1] ** 2))     # 高斯径向基函数
	else:
		raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
	return K

def loadDataSet(filename):
	'''
	:param filename:  文件名
	:return:
		dataMat: 数据矩阵
		labelMat: 类标签
	'''
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():    # 读取数据的每一行
		lineArr = line.strip().split('\t')    # 去除空格，对数据按照特定间隔符进行切割
		dataMat.append([float(lineArr[0]), float(lineArr[1])])    #  数据的前两个特征添加到dataMat中
		labelMat.append(float(lineArr[2]))    #  数据的第三列类标签添加到labelMat中
	return dataMat, labelMat

def calcEk(oS, k):   # 求Ek的误差，预测值-真实值
	'''
	:param oS: optStruct对象
	:param k: 具体的某一行
	:return:
		Ek：预测结果与真实结果对比，计算误差
	'''
	fXk = np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b    # 计算出第k行的预测值
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):     #  随机选择一个整数
	'''
	:param i: 第一个alpha的下标
	:param m: 所有alpha的数目
	:return:
		j: 返回一个不为i的随机数，在0-m之间的整数值
	'''
	j = i
	while j == i:     #  如果i和j相等时，就对j重新赋值一个整数
		j = np.random.randint(0, m-1)
	return j

def selectJ(i, oS, Ei):    #  返回最优的j和Ej
	'''
	:param i: 具体的i行
	:param oS: optStruct对象
	:param Ei: 预测结果与真实结果的对比，计算误差Ei
	:return:
		j: 随机选出的一列
		Ej: 预测结果与真实结果对比，计算误差Ej
	'''
	maxK = -1
	maxDeltaE = 0
	Ej = 0
	oS.eCache[i] = [1, Ei]    #  首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了
	validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]     # 非零E值的行的list列表，所对应的alpha值
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:    #  在所有的值上进行循环，并选择其中使得改变最大的那个值
			if k == i:
				continue
			Ek = calcEk(oS, k)     # 求  Ek误差：预测值-真实值的差
			deltaE = abs(Ei - Ek)
			if deltaE > maxDeltaE:
				#  选择具有最大步长的j
				maxK = k
				maxDeltaE = deltaE
				Ej = Ek
		return maxK, Ej
	else:    #  如果是第一次循环，则随机选择一个alpha值
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)    #  求Ek误差：预测值-真实值的差
	return j, Ej

def updateEk(oS, k):
	'''
	:param oS:  optStruct对象
	:param k:  某一列号
	'''
	Ek = calcEk(oS, k)    # 计算误差：预测值-真实值的差
	oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):    # 调整目标值aj的值，使之处于最小值 L 和最大值 H 之间
	aj = min(aj, H)
	aj = max(L, aj)
	return aj

def inner(i, oS):   # 内循环代码
	'''
	:param i: 具体的某一行
	:param oS: optStruct对象
	:return:
		0 找不到最优的值
		1 找到最优值，并且oS.Cache到缓存中
	'''
	Ei = calcEk(oS, i)    #  计算Ek误差：预测值-真实值的差
	# 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
	# 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
	# 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		# 选择最大的误差对应的j进行优化。效果更明显
		j, Ej = selectJ(i, oS, Ei)
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()
		# L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H:
			return 0
		# eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
		eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
		if eta >= 0:
			print("eta>=0")
			return 0
		# 计算出一个新的alphas[j]值
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
		# 并使用辅助函数，以及L和H对其进行调整
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		updateEk(oS, j)    # 更新误差缓存
		# 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
		if abs(oS.alphas[j] - alphaJold) < 0.00001:
			return 0
		# 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
		updateEk(oS, i)    # 更新误差缓存
		# 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
		# w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
		# 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
		# 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
			oS.b = b2
		else:
			oS.b = (b1 + b2) / 2
		return 1
	else:
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
	'''
	 完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
	:param
		dataMatIn    数据集
		classLabels  类别标签
		C  松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。可以通过调节该参数达到不同的结果。
		toler   容错率
		maxIter 退出前最大的循环次数
		kTup    包含核函数信息的元组
	return:
		b       模型的常量值
		alphas  拉格朗日乘子
	'''
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)    # 创建一个optStruct对象
	iter = 0
	entireSet = True
	alphaPairsChanged = 0
	# 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		# ----------- 第一种写法 start -------------------------
		#  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
		if entireSet:
			# 在数据集上遍历所有可能的alpha
			for i in range(oS.m):
				# 是否存在alpha对，存在就+1
				alphaPairsChanged += inner(i, oS)
				# print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
			iter += 1
		else:    # 对已存在 alpha对，选出非边界的alpha值，进行优化。
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]     # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
			for i in nonBoundIs:
				alphaPairsChanged += inner(i, oS)
				# print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
			iter += 1
		# ----------- 第一种写法 end -------------------------
		# ----------- 第二种方法 start -------------------------
		# if entireSet:   #  遍历整个数据集
		# 	alphaPairsChanged += sum(innerL(i, oS) for i in range(oS.m))
		# else:     #  遍历非边界值
		# 	nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]    # 遍历不在边界0和C的alpha
		# 	alphaPairsChanged += sum(innerL(i, oS) for i in nonBoundIs)
		# ----------- 第二种方法 end -------------------------
		# 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
		iter += 1
		if entireSet:
			entireSet = False  # toggle entire set loop
		elif alphaPairsChanged == 0:
			entireSet = True
		print("iteration number: %d" % iter)
	return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):   # 基于alpha计算w值
	'''
	:param alphas:  拉格朗日乘子
	:param dataArr:  feature数据集
	:param classLabels:  目标变量数据集
	:return:
		wc：回归系数
	'''
	X = np.mat(dataArr)
	labelMat = np.mat(classLabels).T
	m, n = np.shape(X)
	w = np.zeros((n, 1))
	for i in range(m):
		w += np.multiply(alphas[i] * labelMat[i], X[i].T)    # 根据推到，通过alpha的中计算w值
	return w

def testRBF(k1 = 1.3):
	dataArr, labelArr = loadDataSet(r'D:\Data\ML_InAction\SVM\dataSetRBF.txt')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
	datMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A > 0)[0]
	sVs = datMat[svInd]  # get matrix of only support vectors
	labelSV = labelMat[svInd]
	print("there are %d Support Vectors" % np.shape(sVs)[0])
	m, n = np.shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1
	print("the training error rate is: %f" % (float(errorCount) / m))

	dataArr, labelArr = loadDataSet(r'D:\Data\ML_InAction\SVM\dataSetRBF2.txt')
	errorCount = 0
	datMat =np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	m, n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		# 和这个svm-simple类似： fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1
	print("the test error rate is: %f" % (float(errorCount) / m))

def img2Vector(filename):    #  将图片数据转化为数组格式
	returnVec = np.zeros((1, 1024))     # 图片格式的数据位32 * 32 = 1024, 故每个文件均包含1024个数据点，转为一个行向量
	fr = open(filename)
	for i in range(32):
		linstr = fr.readline()
		for j in range(32):
			returnVec[0, 32 * i + j] = int(linstr[j])    #  数组对应位置为图片矩阵上的点
	return returnVec

def loadImages(dirname):
	import os
	hwLabels = []
	print(dirname)
	trainingFileList = os.listdir(dirname)    #   当前目录下，所有的文件列表
	m = len(trainingFileList)   #  文件数量
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.strip('.')[0]  # 文件格式：1_0.txt, 1_1.txt, 1_2.txt, 9_0.txt, 9_1.txt, 9_13.txt等
		classNumStr = int(fileStr.split('_')[0])
		if classNumStr == 9:    #  本例使用的数据仅包含9和1两个数字的识别
			hwLabels.append(-1)
		else:
			hwLabels.append(1)
		trainingMat[i, :] = img2Vector('%s/%s' % (dirname, fileNameStr))
	return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
	# 1. 导入训练数据
	dataArr, labelArr = loadImages('文件路径')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
	datMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()    #  矩阵的转置，效果等同于dataMat.T
	svInd = np.nonzero(alphas.A > 0)[0]
	sVs = datMat[svInd]
	labelSV = labelMat[svInd]
	# print("there are %d Support Vectors" % shape(sVs)[0])
	m, n = np.shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print("the training error rate is: %f" % (float(errorCount) / m))
	# 2. 导入测试数据
	dataArr, labelArr = loadImages('测试数据文件')
	errorCount = 0
	datMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	m, n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print("the test error rate is: %f" % (float(errorCount) / m))

def plotfig_SVM(xArr, yArr, ws, b, alphas):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	b = np.array(b)[0]    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])    # 注意flatten的用法
	x = np.arange(-1.0, 10.0, 0.1)    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
	y = (-b - ws[0, 0] * x) / ws[1, 0]    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
	ax.plot(x, y)
	for i in range(np.shape(yMat[0, :])[1]):
		if yMat[0, i] > 0:
			ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
		else:
			ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
	# 找到支持向量，并在图中标红
	for i in range(100):
		if alphas[i] > 0.0:
			ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
	plt.show()


if __name__ == "__main__":
	# 无核函数的测试
	# 获取特征和目标变量
	dataArr, labelArr = loadDataSet(r'D:\Data\ML_InAction\SVM\dataSet.txt')
	# print(labelArr)

	# b是常量值，alphas是拉格朗日乘子
	b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
	print('/n/n/n')
	print('b=', b)
	print('alphas[alphas>0]=', alphas[alphas > 0])
	print('shape(alphas[alphas > 0])=', np.shape(alphas[alphas > 0]))
	for i in range(100):
		if alphas[i] > 0:
			print(dataArr[i], labelArr[i])
	# 画图
	ws = calcWs(alphas, dataArr, labelArr)
	plotfig_SVM(dataArr, labelArr, ws, b, alphas)

	#有核函数的测试
	testRBF(0.8)

	# 项目实战
	# 示例：手写识别问题回顾
	# testDigits(('rbf', 0.1))
	# testDigits(('rbf', 5))
	#testDigits(('rbf', 10))
	# testDigits(('rbf', 50))
	# testDigits(('rbf', 100))
	# testDigits(('lin', 10))