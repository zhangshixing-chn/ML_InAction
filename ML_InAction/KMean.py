#********========KMeans：聚类算法(sklearn)========********#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
from sklearn.cluster import KMeans
# dataset = []
# fr = open(r'D:\Data\ML_InAction\KMeans\testSet.txt')
# for line in fr.readlines():
# 	curLine = line.strip().split('\t')
# 	#flLine = list(map(float, curLine))
# 	flLine = [float(i) for i in curLine]
# 	dataset.append(flLine)
# km = KMeans(n_clusters = 4)
# km.fit(dataset)
# km_Pred = km.predict(dataset)
# print(km_Pred)
# centers = km.cluster_centers_   # KMeans的聚类中心点
# plt.scatter(np.array(dataset)[:, 1], np.array(dataset)[:, 0], c = km_Pred)
# plt.scatter(centers[:, 1], centers[:, 0], c = "r")
# plt.show()

#********========KMeans：案例实践=========********#
def loadDataSet(filename):   # 读取数据集
	'''
	:param filename: 需要加载的数据文件
	:return:
		返回一个矩阵格式的数据
	'''
	dataset = []    # 初始化一个空列表
	fr = open(filename)    # 读取文件
	for line in fr.readlines():    # 循环遍历文件所有行
		curLine = line.strip().split('\t')    # 切割每一行的数据
		#flLine = [float(i) for i in curLine]    # 将数据转换为浮点类型,便于后面的计算
		flLine = list(map(float, curLine))     # 映射所有的元素为 float（浮点数）类型
		dataset.append(flLine)     # 将数据追加到dataset
	return np.mat(dataset)    # 返回dataset

def distEclud(vecA, vecB):   # 计算点到聚类中心点的距离
	dist = np.sqrt(np.sum(np.power(vecA - vecB, 2)))    # 欧几里得距离(欧氏距离)
	return dist

def randCent(dataset, k):    # 产生k个随机的初始化聚类中心点，数值大小在每一列的最值范围内
	'''
	为给定数据集构建一个包含K个随机质心的集合,随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成,
	然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
	:param dataset: 数据集
	:param k: 产生k个初始化聚类中心点
	:return:  k个聚类中心点列表
	'''
	n = np.shape(dataset)[1]    # 获取数据集的特征个数
	# print(type(dataset))
	centroids = np.mat(np.zeros((k, n)))    # 初始化质心,创建(k,n)个以零填充的矩阵
	for i in range(n):    # 循环遍历特征值
		minI = np.min(dataset[:, i])    # 计算每一列的最小值
		maxI = np.max(dataset[:, i])    # 计算每一列的最大值
		rangeI = float(maxI - minI)     # 计算每一列的范围值
		centroids[:, i] = np.mat(minI + rangeI * np.random.rand(k, 1))    # 计算每一列的质心,并将值赋给centroids
	return centroids     # 返回质心

def kMeans(dataset, k, disMeas=distEclud, createCent=randCent):
	'''
	 创建K个质心,然后将每个店分配到最近的质心,再重新计算质心。这个过程重复数次,直到数据点的簇分配结果不再改变为止
	:param dataset: 聚类数据集
	:param k: 簇的数目
	:param disMeas: 计算距离的函数
	:param createCent: 创建初始质心
	:return:
	'''
	m, n = np.shape(dataset)    # 数据集的样本数和特征数
	clusterAssment = np.mat(np.zeros((m, 2)))   # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离)
	centroids = createCent(dataset, k)   # 创建质心,随机K个质心
	clusterChanged = True     # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
	while clusterChanged:
		clusterChanged = False
		# 遍历所有数据找到距离每个点最近的质心,
		# 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
		for i in range(m):
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				# 计算数据点到质心的距离
				# 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
				disJI = disMeas(centroids[j, :], dataset[i, :])    # 计算当前点到不同聚类中心点的距离
				# 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
				if disJI < minDist:
					minDist = disJI
					minIndex = j    # 更新最小聚类中心的索引
			# 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True    # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
			clusterAssment[i, :] = minIndex, minDist ** 2    # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
		# print(centroids)
		# print(clusterAssment)
		for cent in range(k):    # 遍历所有质心并更新它们的取值
			# 通过数据过滤来获得给定簇的所有点
			ptsInClust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
			# 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
			centroids[cent, :] = np.mean(ptsInClust, axis=0)
	# 返回所有的类质心与点分配结果
	return centroids, clusterAssment

def biKMeans(dataset, k, distMeas = distEclud):    # 在给定数据集,所期望的簇数目和距离计算方法的条件下,函数返回聚类结果
	'''
	:param dataset: 输入数据集
	:param k: 聚类中心个数
	:param distMeas: 距离计算函数
	:return:
	'''
	m, n = np.shape(dataset)
	clusterAssment = np.mat(np.zeros((m, 2)))   # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
	centroid0 = np.mean(dataset, axis = 0).tolist()[0]    # 计算整个数据集的质心,并使用一个列表来保留所有的质心[-0.15772275000000002, 1.2253301166666664]
	print('初始聚类中心为：',centroid0)
	centList = [centroid0]
	for j in range(m):   # 遍历数据集中所有点来计算每个点到质心的距离（误差值）
		clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataset[j, :]) ** 2   # 平方误差的大小
	while (len(centList) < k):    # 对簇不停的进行划分,直到得到想要的簇数目为止
		lowestSSE = np.inf    # 初始化最小SSE为无穷大,用于比较划分前后的SSE
		for i in range(len(centList)):  # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
			ptsInCurrCluster = dataset[np.nonzero(clusterAssment[:, 0].A == i)[0], :]   # 对每一个簇,将该簇中的所有点堪称一个小的数据集
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)   # kMeans会生成两个质心(簇),同时给出每个簇的误差值
			sseSplit = sum(splitClustAss[:, 1])   # 将误差值与剩余数据集的误差之和作为本次划分的误差
			sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
			print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE:    # 如果本次划分的SSE值最小,则本次划分被保存
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
		# 更新为最佳聚类中心
		bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		print('the bestCentToSplit is: ', bestCentToSplit)
		print('the len of bestClustAss is: ', len(bestClustAss))
		# 更新质心列表
		# 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
		centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
		# 添加bestNewCents的第二个质心
		centList.append(bestNewCents[1, :].tolist()[0])
		# 重新分配最好簇下的数据(质心)以及SSE
		clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	return np.mat(centList), clusterAssment

if __name__ == '__main__':
	filename1 = r'D:\Data\ML_InAction\KMeans\testSet.txt'
	filename2 = r'D:\Data\ML_InAction\KMeans\testSet2.txt'
	dataMat = loadDataSet(filename2)
	biKMeans(dataMat, 3)