#********========SVD: 奇异值分解算法========********#
import numpy as np

def loadExData1():
	# 推荐引擎示例矩阵
	# return[[4, 4, 0, 2, 2], [4, 0, 0, 3, 3], [4, 0, 0, 1, 1], [1, 1, 1, 2, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0]]
	# return[[1, 1, 1, 0, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0], [1, 1, 0, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1]]
	return [[0, -1.6, 0.6],
	        [0, 1.2, 0.8],
	        [0, 0, 0],
	        [0, 0, 0]]

def loadExData2():
	return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
	        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
	        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
	        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
	        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
	        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
	        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
	        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
	        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
	        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
	        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def loadExData3():
	return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
	        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
	        [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
	        [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
	        [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
	        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
	        [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
	        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
	        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
	        [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
	        [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

def ecludSim(inA, inB):    # 欧氏距离
	return 1.0 / (1.0 + np.linalg.norm(inA - inB))    # 利用numpy中的lialg.norm函数计算矩阵或者向量的范数

def pearsSim(inA, inB):    # 皮尔逊相关系数距离
	if len(inA) < 3:    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
		return 1.0
	return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):   # 计算余弦相似度距离, 如果夹角为90度，相似度为0；如果两个向量的方向相同，相似度为1.0
	num = float(inA.T * inB)
	denom = np.linalg.norm(inA) * np.linalg.norm(inB)
	return 0.5 + 0.5 * (num / denom)

# 基于物品相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
	"""standEst(计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分)
	Args:
	    dataMat         训练数据集
	    user            用户编号
	    simMeas         相似度计算方法
	    item            未评分的物品编号
	Returns:
	    ratSimTotal/simTotal     评分（0～5之间的值）
	"""
	n = np.shape(dataMat)[1]  # 得到数据集中的物品数目（数据矩阵的列数）
	# 初始化两个评分值
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
		userRating = dataMat[user, j]    # user 对 j 物品的评分
		if userRating == 0:     # 如果某个物品的评分值为0，则跳过这个物品
			continue
		# 寻找两个用户都评级的物品
		# 变量 overLap 给出的是两个物品当中已经被评分的那个元素的索引ID
		# logical_and(x1, x2) 表示计算x1 和 x2元素的真值
		overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 返回数组索引值
		# overLap 给出的是两个物品当中已经被评分的那个元素的索引ID
		if len(overLap) == 0:    # 如果相似度为0，则两者没有任何重合元素，终止本次循环
			similarity = 0
		# 如果存在重合的物品，则基于这些重合物重新计算相似度
		else:
			similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
		# print('the %d and %d similarity is : %f'(item,j,similarity))
		# 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
		# similarity  用户相似度，   userRating 用户评分
		simTotal += similarity    # 不断累加所有物品与给定item之间的相似度
		ratSimTotal += similarity * userRating    # similarity  用户相似度，    userRating 用户评分
	if simTotal == 0:
		return 0
	# 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
	else:
		return ratSimTotal / simTotal    # 对最后的评分数据转化为特定区间范围内

# 基于SVD的评分估计
# 在recommend() 中，这个函数用于替换对standEst()的调用，该函数对给定用户给定物品构建了一个评分估计值
def svdEst(dataMat, user, simMeas, item):
	"""svdEst( )
	Args:
	    dataMat         训练数据集
	    user            用户编号
	    simMeas         相似度计算方法
	    item            未评分的物品编号
	Returns:
	    ratSimTotal/simTotal     评分（0～5之间的值）
"""
	n = np.shape(dataMat)[1]    # 物品数目
	simTotal = 0.0
	ratSimTotal = 0.0
	U, Sigma, VT = np.linalg.svd(dataMat)    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
	# # 分析 Sigma 的长度取值
	# analyse_data(Sigma, 20)
	# 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
	Sig4 = np.mat(np.eye(4) * Sigma[: 4])
	xformedItems = dataMat.T * U[:, :4] * Sig4.I    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
	print('dataMat', np.shape(dataMat))
	print('U[:, :4]', np.shape(U[:, :4]))
	print('Sig4.I', np.shape(Sig4.I))
	print('VT[:4, :]', np.shape(VT[:4, :]))
	print('xformedItems', np.shape(xformedItems))
	# 这和standEst()函数中的for循环的目的一样，只不过这里的相似度计算时在低维空间下进行的
	for j in range(n):    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
		userRating = dataMat[user, j]
		if userRating == 0 or j == item:
			continue
		similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)    # 降维之后，已经确保每一列均有值
		print('the %d and %d similarity is: %f' % (item, j, similarity))
		simTotal += similarity    # 对相似度不断累加求和
		ratSimTotal += similarity * userRating    # 对相似度及对应评分值的乘积求和
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal/simTotal    # 计算估计评分

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
	# recommend()函数，就是推荐引擎，它默认调用standEst()函数，产生了最高的N个推荐结果。
	# 如果不指定N的大小，则默认值为3。该函数另外的参数还包括相似度计算方法和估计方法
	"""svdEst( )
	    Args:
	        dataMat         训练数据集
	        user            用户编号
	        simMeas         相似度计算方法
	        estMethod       使用的推荐算法
	    Returns:
	        返回最终 N 个推荐结果
	"""
	unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]  # 寻找未评级的物品；对给定用户建立未评分的物品列表
	if len(unratedItems) == 0:    # 如果不存在未评分物品，那么就退出函数
		return 'you rated everything'
	itemScores = []   # 物品的编号和评分值
	# 在未评分物品上进行循环
	for item in unratedItems:
		# 获取 item 该物品的评分
		estimatedScore = estMethod(dataMat, user, simMeas, item)    # 获取item该物品的评分
		itemScores.append((item, estimatedScore))
	return sorted(itemScores, key = lambda a: a[1], reverse=True)[:N]   # 对评分从大到小排序，去排前N个未评级的物品作为推荐结果

def analyse_data(sigma, loopNum=20):
	"""analyse_data(分析 sigma 的长度取值)
	   Args:
	       sigma         sigma的值
	       loopNum       循环次数
	"""
	sig2 = sigma ** 2    # 总方差的集合（总能量值）
	sigmasum = sum(sig2)
	for i in range(loopNum):
		sigmaI = sum(sig2[:i+1])
		print('主成分：%s, 方差占比：%s%%' % (format(i + 1, '2.0f'), format(sigmaI / sigmasum * 100, '4.2f')))

def imgLoadData(filename):   # 图像压缩函数，加载并转换数据
	data = []
	fr = open(filename)
	for line in fr.readlines():   # 对于每一行数据
		newRow = []
		for i in range(32):   # 因为图片的像素点为32*32
			newRow.append(int(line[i]))
		data.append(newRow)
	dataMat = np.mat(data)
	return dataMat

def printMat(inMat, thresh=0.8):    # 打印矩阵
	# 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
	for i in range(32):
		for j in range(32):
			if float(inMat[i, j]) > thresh:
				print(1, end='')
			else:
				print(0, end='')
		print('')

def imgCompress(numSV=3, thresh=0.8):    # 实现图像压缩，允许基于任意给定的奇异值数目重构图像
	"""imgCompress( )
	   Args:
	       numSV       Sigma长度
	       thresh      判断的阈值
	"""
	myMat = imgLoadData(r'D:\Data\ML_InAction\SVD\0_5.txt')     # 构建一个列表
	print("****original matrix****")
	printMat(myMat, thresh)     # 原始图像
	# 通过Sigma 重新构成SigRecom来实现
	# Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
	U, sigma, VT = np.linalg.svd(myMat)
	SigRecon = np.mat(np.zeros((numSV, numSV)))
	for k in range(numSV):
		SigRecon[k, k] = sigma[k]
	analyse_data(sigma, 20)     # 分析插入的 Sigma 长度
	SigRecon = np.mat(np.eye(numSV) * sigma[: numSV])
	reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]     # 重构的数据
	print("****reconstructed matrix using %d singular values *****" % numSV)
	printMat(reconMat, thresh)

if __name__ == '__main__':
	# # 对矩阵进行SVD分解(用python实现SVD)
	# Data = loadExData1()
	# print('Data:', Data)
	# U, Sigma, VT = np.linalg.svd(Data)
	# # 打印Sigma的结果，因为前3个数值比其他的值大了很多，为9.72140007e+00，5.29397912e+00，6.84226362e-01
	# # 后两个值比较小，每台机器输出结果可能有不同可以将这两个值去掉
	# print('U:', U)
	# print('Sigma', Sigma)
	# print('VT:', VT)
	# print('VT:', VT.T)

	# 计算欧氏距离
	# myMat = np.mat(loadExData3())
	# # print(myMat)
	# print(ecludSim(myMat[:, 0], myMat[:, 4]))
	# print(ecludSim(myMat[:, 0], myMat[:, 0]))

	# # 计算余弦相似度
	# print(cosSim(myMat[:, 0], myMat[:, 4]))
	# print(cosSim(myMat[:, 0], myMat[:, 0]))

	# # 计算皮尔逊相关系数
	# print(pearsSim(myMat[:, 0], myMat[:, 4]))
	# print(pearsSim(myMat[:, 0], myMat[:, 0]))

	# 计算相似度的方法
	# myMat = np.mat(loadExData3())
	# # print(myMat)

	# # 计算相似度的第一种方式
	# print(recommend(myMat, 1, estMethod=svdEst))

	# # 计算相似度的第二种方式
	# print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

	# # 默认推荐（菜馆菜肴推荐示例）
	# print(recommend(myMat, 2))
	# 利用SVD提高推荐效果
	# U, Sigma, VT = np.linalg.svd(np.mat(loadExData2()))
	# print(Sigma)                 # 计算矩阵的SVD来了解其需要多少维的特征
	# Sig2 = Sigma**2              # 计算需要多少个奇异值能达到总能量的90%
	# print(sum(Sig2))             # 计算总能量
	# print(sum(Sig2) * 0.9)       # 计算总能量的90%
	# print(sum(Sig2[: 2]))        # 计算前两个元素所包含的能量
	# print(sum(Sig2[: 3]))        # 两个元素的能量值小于总能量的90%，于是计算前三个元素所包含的能量
	imgCompress(2)