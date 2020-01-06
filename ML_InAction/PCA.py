#********========PCA: 主成分分析案例========********#
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False     # 用来正常显示负号

def loadDataSet(filename, delim = '\t'):
	#dataMat = []
	fr = open(filename)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	dataMat = [list(map(float, line)) for line in stringArr]  # python 3 需要在map()函数之前加上list(),否则只以内存方式存储
	# for line in fr.readlines():
	# 	curLine = line.strip().split(delim)
	# 	lineArr = [float(i) for i in curLine]
	# 	# lineArr = []
	# 	# for i in curLine:
	# 	# 	lineArr.append(float(i))
	# 	dataMat.append(lineArr)
	return np.mat(dataMat)

def pca(dataset, topNfeat = 999999):
	"""
	:param dataset: 原数据集矩阵
	:param topNfeat: 应用的N个特征
	:return:
		lowDataMat: 降维后数据集
		reconMat: 新的数据集空间
	"""
	meanVals = np.mean(dataset, axis = 0)    # 计算每一列的均值
	# print('meansVals:', meanVals)
	meanRemobed = dataset - meanVals    # 每个向量同时都减去 均值
	# print('meanRemobed:', meanRemobed)
	'''
	    方差：（一维）度量两个随机变量关系的统计量
	    协方差： （二维）度量各个维度偏离其均值的程度
	    协方差矩阵：（多维）度量各个维度偏离其均值的程度
	    当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
	    当 cov(X, Y)<0时，表明X与Y负相关；
	    当 cov(X, Y)=0时，表明X与Y不相关。
	'''
	covMat = np.cov(meanRemobed, rowvar=0)     # 求协方差矩阵,cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))   # 求协方差矩阵的特征值和特征向量
	# print('eigVals=', eigVals)
	# print('eigVects=', eigVects)
	eigValInd = np.argsort(eigVals)            # 对特征值从小到大排列，返回从小到大的index索引值
	# print('eigValInd1=', eigValInd)
	eigValInd = eigValInd[:(-topNfeat+1):-1]   # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
	# print('eigValInd2=', eigValInd)
	redEigVects = eigVects[:, eigValInd]       # 返回特征向量按照特征值从大到小的值， 重组 eigVects 最大到最小
	# print('redEigVects=', redEigVects.T)
	lowDataMat = meanRemobed * redEigVects     # 将数据转换到新空间
	# print( "---", shape(meanRemoved), shape(redEigVects))
	reconMat = (lowDataMat * redEigVects.T) + meanVals
	# print('lowDDataMat=', lowDDataMat)
	# print('reconMat=', reconMat)
	return redEigVects, reconMat

def replaceNanWithMean():    # 缺失值填充及处理过程
	dataset = loadDataSet(r'D:\Data\ML_InAction\PCA\secom.data', ' ')
	numFeat = np.shape(dataset)[1]   #  数据集包含的特征数量
	for i in range(numFeat):
		meanVal = np.mean(dataset[np.nonzero(~ np.isnan(dataset[:, i].A))[0], i])   # 计算特征列非空值数据的均值，.A 返回矩阵基于的数组
		dataset[np.nonzero(np.isnan(dataset[:, i].A))[0], i] = meanVal   # 替换特征列空值位置的数据
	return dataset

def show_picture(dataset, reconMat, cov_score_precent):
	fig = plt.figure(figsize=(16, 6))
	ax1 = fig.add_subplot(131)
	ax1.scatter(dataset[:, 0].flatten().A[0], dataset[:, 1].flatten().A[0], marker='^', s=90)
	ax2 = fig.add_subplot(132)
	ax2.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
	ax3 = fig.add_subplot(133)
	ax3.plot(cov_score_precent, 'b-*')
	ax3.set_ylabel('方差的百分比')
	ax3.set_xlabel('主成分数目')
	plt.show()

def analyse_data(dataset):
	meanVals = np.mean(dataset, axis=0)    # 数据集的列均值
	meanRemoved = dataset - meanVals
	covMat = np.cov(meanRemoved, rowvar=0)
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))    # 求解特征值与特征向量
	eigValInd = np.argsort(eigVals)    # 对特征值进行排序，获取从小到大的索引值

	topNfeat = 20    # 选取的特征个数
	eigValInd = eigValInd[:-(topNfeat + 1):-1]
	cov_all_score = float(sum(eigVals))
	sum_cov_score = 0
	cov_score_precent = []
	for i in range(0, len(eigValInd)):
		line_cov_score = float(eigVals[eigValInd[i]])
		sum_cov_score += line_cov_score
		print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i + 1, '2.0f'), format(line_cov_score / cov_all_score * 100, '4.2f'),
			format(sum_cov_score / cov_all_score * 100, '4.1f')))
		cov_score_precent.append(line_cov_score / cov_all_score * 100)
	return cov_score_precent, sum_cov_score

if __name__ == '__main__':
	dataMat = replaceNanWithMean()
	print(np.shape(dataMat))
	cov_score_precent, _ = analyse_data(dataMat)
	redEigVects, reconMat = pca(dataMat)
	show_picture(dataMat, reconMat, cov_score_precent)