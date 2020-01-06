'''
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection    # LineCollection实现在图形中绘制多条线
# from sklearn.linear_model import LinearRegression
# from sklearn.isotonic import IsotonicRegression    # 等式回归
# from sklearn.utils import check_random_state
#
# n = 100
# x = np.arange(n)
# rs = check_random_state(0)
# y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
# ir = IsotonicRegression()
# y_ = ir.fit_transform(x, y)
# lr = LinearRegression()
# lr.fit(x[:, np.newaxis], y)  # 线性回归的 x 需要为 2d   # np.newaxis的作用就是选取部分的数据增加一个维度
#
# segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
# lc = LineCollection(segments, zorder=0)
# lc.set_array(np.ones(len(y)))
# lc.set_linewidths(0.5 * np.ones(n))
#
# fig = plt.figure()
# plt.plot(x, y, 'r.', markersize=12)
# plt.plot(x, y_, 'g.-', markersize=12)
# plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
# plt.gca().add_collection(lc)
# plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
# plt.title('Isotonic regression')
# plt.show()

# Kernel ridge regression ( 内核岭回归 )
# 2.1 Comparison of kernel ridge regression and SVR (内核岭回归与 SVR 的比较)
from __future__ import division
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)
# 生成样本数据
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()
# 给目标增加噪音
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
X_plot = np.linspace(0, 5, 100000)[:, None]
# Fit regression model ( 拟合 回归 模型 )
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)
t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)
sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)
t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (X_plot.shape[0], svr_predict))
t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (X_plot.shape[0], kr_predict))
# 查看结果
sv_ind = svr.best_estimator_.support_
plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors', zorder=2)
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1)
#plt.hold('on')
plt.plot(X_plot, y_svr, c='r', label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.plot(X_plot, y_kr, c='g', label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
# 可视化训练和预测时间
plt.figure()
# 生成样本数据
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
sizes = np.logspace(1, 4, 7, dtype=np.int)
for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1, gamma=10), "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:train_test_size], y[:train_test_size])
        train_time.append(time.time() - t0)
        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)
    plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g", label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g", label="%s (test)" % name)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
# 可视化学习曲线
plt.figure()
svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10), scoring="neg_mean_squared_error", cv=10)
train_sizes_abs, train_scores_kr, test_scores_kr = learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10), scoring="neg_mean_squared_error", cv=10)
plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r", label="SVR")
plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g", label="KRR")
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
plt.show()
'''

#========********Regression：回归案例********=========#
import time
import bs4
import json
import requests
import urllib3
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from bs4 import BeautifulSoup

def loadDataSet(filename):
    '''
    :param filename: 加载文件内容
    :return:
        dataMat: feature对应的数据
        labelMat: feature对应的分类标签，即类别标签
    '''
    numFeat = len(open(filename).readline().split('\t')) - 1   # 获取样本特征数量，不计算最后的目标变量
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():    # 读取每一行记录
        lineArr = []
        curLine = line.strip().split('\t')   # 删除一行中以tab分隔的数据前后的空白符号
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))   # 将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
        dataMat.append(lineArr)     # 将测试数据的输入数据部分存储到dataMat 的List中
        labelMat.append(float(curLine[-1]))     # 将每一行的最后一个数据，即类别，或者叫目标变量存储到labelMat List中
    return dataMat, labelMat

def standRegres(xArr, yArr):    # 标准回归，回归系数的估计函数，输入为：特征矩阵X和目标值Y
    '''
    :param xArr: 输入的样本数据，包含每个样本数据的 feature
    :param yArr: 对应于输入数据的类别标签，也就是每个样本对应的目标变量
    :return:
        ws: 回归系数
    '''
    xMat = np.mat(xArr)    # mat()函数将xArr，yArr转换为矩阵
    yMat = np.mat(yArr).T    # mat().T 代表的是对矩阵进行转置操作
    xTx = xMat.T * xMat    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    if np.linalg.det(xTx) == 0.0:    # 判断矩阵行列式是否0，矩阵是否可逆，如果行列式为0，则矩阵不可逆
        print('This Matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)    #  利用最小二乘法求解权重系数，其中xTx.I表示矩阵的逆，与inv()函数作用一样
    return ws

def lwlr(testpoint, xArr, yArr, k = 1.0):     # 局部加权线性回归,在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归
    '''
    :param testpoint: 样本点
    :param xArr: 样本特征数据，即feature
    :param yArr: 每个样本分类标签，即目标变量
    :param k: 赋予权重矩阵的核的一个参数，与权重的衰减速率有关
    :return:
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]   # xMat矩阵的行数
    weights = np.mat(np.eye(m))   # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    for i in range(m):
        diffMat = testpoint - xMat[i, :]    #  testPoint 的形式是 一个行向量的形式，计算测试点与xMat矩阵每一行样本点之间的距离
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))   # 通过高斯核函数计算测试点与样本点的权重值。 # k控制衰减的速度
    xTx = xMat.T * (weights * xMat)    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    if np.linalg.det(xTx) == 0.0:
        print('This Matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))    # 计算出回归系数的一个估计
    return testpoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):    # 测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
    '''
    :param testArr: 测试所用的所有样本点
    :param xArr: 样本的特征数据，即 feature
    :param yArr: 每个样本对应的类别标签，即目标变量
    :param k: 控制核函数的衰减速率
    :return:
         yHat：预测点的估计值
    '''
    m = np.shape(testArr)[0]   # 样本点总数
    yHat = np.zeros(m)     # 构建一个全部都是 0 的 1 * m 的矩阵
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)    # 循环所有的数据点，并将lwlr运用于所有的数据点
    return yHat    # 返回估计值

def lwlrTestPlot(xArr, yArr, k =1.0):    # 先对xArr排序，函数返回样本点的估计值和实际值
    '''
    :param xArr: 样本的特征数据，即 feature
    :param yArr: 每个样本对应的类别标签，即目标变量，实际值
    :param k: 控制核函数的衰减速率的有关参数，这里设定的是常量值 1
    :return:
        yHat：样本点的估计值
        xCopy：xArr的复制
    '''
    yHat = np.zeros(np.shape(yArr))   # 生成一个与目标变量数目相同的 0 向量
    xCopy = np.mat(xArr)    # xArr转化为矩阵形式
    xCopy.sort(0)    # 排序
    for i in range(np.shape(xArr)[0]):    # 开始循环，为每个样本点进行局部加权线性回归，得到最终的目标变量估计值
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy

def rssError(yArr, yHat):   # 统计汇总实际值与估计值之间的预测误差大小
    '''
    :param yArr: 真实的目标变量
    :param yHat: 预测得到的估计值
    :return:
         计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    err_sum = ((yArr - yHat) ** 2).sum()
    return err_sum

def ridgeRegression(xMat, yMat, lam = 0.2):   # 岭回归，解决特征矩阵奇异问题，或者特征变量多于样本数据的数据集
    '''
    :param xMat: 样本的特征数据，即 feature
    :param yMat: 每个样本对应的类别标签，即目标变量，实际值
    :param lam: 引入的一个λ值，使得矩阵非奇异
    :return:
        经过岭回归公式计算得到的回归系数
    '''
    xTx = xMat.T * xMat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    demon = xTx + np.eye(np.shape(xMat)[1]) * lam    # 特征矩阵的变体，增加lam * I
    if np.linalg.det(demon) == 0.0:
        print('This Matrix is singular, cannot do inverse')
        return
    ws = demon.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):    # 函数 ridgeTest() 用于在一组 λ 上测试结果
    '''
    :param xArr: 样本数据的特征，即 feature
    :param yArr: 样本数据的类别标签，即真实数据
    :return:
        wMat：将所有的回归系数输出到一个矩阵并返回
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)    # 计算Y的均值，axis=0表示求列均值
    xMean = np.mean(xMat, 0)    # 计算X的列均值
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar   # xMat的标准化
    numTestPts = 30    # 可以在 30 个不同的 lambda 下调用 ridgeRegres() 函数
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))    # 创建30 * m 的全部数据为0 的矩阵
    for i in range(numTestPts):
        ws = ridgeRegression(xMat, yMat, np.exp(i-10))    # exp() 返回 e^x
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):    # 标准化
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)    # 计算平均值然后减去它
    inVar = np.var(inMat, 0)    # 计算除以Xi的方差
    inMat = (inMat - inMeans) / inVar
    return inMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):   # 前向逐步回归
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean    # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        #print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)    #  np.mat().A表示将矩阵转化为array数组类型
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

# def scrapePage(inFile, outFile, yr, numPce, origPrc):
#     fr = open(inFile)
#     fw = open(outFile, 'a')     # a is append mode writing
#     soup = BeautifulSoup(fr.read())
#     i = 1
#     currentRow = soup.findAll('table', r="%d" % i)
#     while (len(currentRow) != 0):
#         title = currentRow[0].findAll('a')[1].text
#         lwrTitle = title.lower()
#         if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#             newFlag = 1.0
#         else:
#             newFlag = 0.0
#         soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#         if len(soldUnicde) == 0:
#             print("item #%d did not sell" % i)
#         else:
#             soldPrice = currentRow[0].findAll('td')[4]
#             priceStr = soldPrice.text
#             priceStr = priceStr.replace('$', '')  # strips out $
#             priceStr = priceStr.replace(',', '')  # strips out ,
#             if len(soldPrice) > 1:
#                 priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
#             print("%s\t%d\t%s" % (priceStr, newFlag, title))
#             fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
#         i += 1
#         currentRow = soup.findAll('table', r="%d" % i)
#     fw.close()

# 预测乐高玩具套装的价格 ------ 最初的版本，因为现在 google 的 api 变化，无法获取数据
# 故改为了下边的样子，但是需要安装一个 beautifulSoup 这个第三方网页文本解析器，安装很简单，见下边
# from time import sleep
# import json
# 这里特别指出 正确的使用方法为下面的语句使用,from urllib import request 将会报错,具体细节查看官方文档
# import urllib.request   # 在Python3中将urllib2和urllib等五个模块合并为一个标准库urllib,其中的urllib2.urlopen更改为urllib.request.urlopen

# def searchForSet(retX, retY, setNum, yr, numPce, origPrc):     # 预测乐高玩具套装价格的函数
#     time.sleep(5)
#     myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#     searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#     http = urllib3.PoolManager()
#     pg = http.request('GET', searchURL)
#     retDict = json.loads(pg.read())  # 转换为json格式
#     for i in range(len(retDict['items'])):
#         try:
#             currItem = retDict['items'][i]
#             if currItem['product']['condition'] == 'new':
#                 newFlag = 1
#             else:
#                 newFlag = 0
#             listOfInv = currItem['product']['inventories']
#             for item in listOfInv:
#                 sellingPrice = item['price']
#                 if sellingPrice > origPrc * 0.5:
#                     print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
#                     retX.append([yr, numPce, newFlag, origPrc])
#                     retY.append(sellingPrice)
#         except:
#             print('problem with item %d' % i)
#
# def setDataCollect(retX, retY):
#     searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#     searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#     searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#     searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#     searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#     searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
#
# def crossValidation(xArr, yArr, numVal = 10):     # 交叉验证岭回归函数
#     m = len(yArr)
#     indexList = range(m)
#     errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows创建error mat 30columns numVal 行
#     for i in range(numVal):
#         trainX = []
#         trainY = []
#         testX = []
#         testY = []
#         np.random.shuffle(indexList)
#         for j in range(m):
#             # 基于indexList中的前90%的值创建训练集
#             if j < m * 0.9:
#                 trainX.append(xArr[indexList[j]])
#                 trainY.append(yArr[indexList[j]])
#             else:
#                 testX.append(xArr[indexList[j]])
#                 testY.append(yArr[indexList[j]])
#         wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
#         for k in range(30):  # loop over all of the ridge estimates
#             matTestX = np.mat(testX)
#             matTrainX = np.mat(trainX)
#             meanTrain = np.mean(matTrainX, 0)
#             varTrain = np.var(matTrainX, 0)
#             matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
#             yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # test ridge results and store
#             errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
#             # print (errorMat[i,k])
#     meanErrors = np.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
#     minMean = float(min(meanErrors))
#     bestWeights = wMat[np.nonzero(meanErrors == minMean)]
#     # can unregularize to get model
#     # when we regularized we wrote Xreg = (x-meanX)/var(x)
#     # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
#     xMat = np.mat(xArr)
#     yMat = np.mat(yArr).T
#     meanX = np.mean(xMat, 0)
#     varX = np.var(xMat, 0)
#     unReg = bestWeights / varX
#     print("the best model from Ridge Regression is:\n", unReg)
#     print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))

# ----------------------------------------------------------------------------
# 预测乐高玩具套装的价格 可运行版本，我们把乐高数据存储到了我们的 input 文件夹下，使用 urllib爬取,bs4解析内容
# 前提：安装 BeautifulSoup，步骤如下
# 在这个页面 https://www.crummy.com/software/BeautifulSoup/bs4/download/4.4/ 下载，beautifulsoup4-4.4.1.tar.gz
# 将下载文件解压，使用 windows 版本的 cmd 命令行，进入解压的包，输入以下两行命令即可完成安装
# python setup.py build
# python setup.py install
# 如果为linux或者mac系统可以直接使用pip进行安装 pip3 install bs4
# ----------------------------------------------------------------------------

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):    # 从页面读取数据，生成retX和retY列表
    # 打开并读取HTML文件
    fr = open(inFile)    # 这里推荐使用with open() 生成器,这样节省内存也可以避免最后忘记关闭文件的问题
    soup = BeautifulSoup(fr.read())
    i=1
    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r = "%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.findAll('table', r = "%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print ("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'data/8.Regression/setHtml/lego10196.html', 2009, 3263, 249.99)

def crossValidation(xArr, yArr, numVal = 10):     # 交叉验证测试岭回归
    m = len(yArr)    # 获得数据点个数，xArr和yArr具有相同长度
    indexList = range(m)
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):    # 主循环 交叉验证循环
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        trainX = []
        trainY = []
        testX = []
        testY = []
        np.random.shuffle(indexList)    # 对数据进行混洗操作
        for j in range(m):    # 切分训练集和测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)    # 获得回归系数矩阵
        for k in range(30):    # 循环遍历矩阵中的30组回归系数
            # 读取训练集和数据集
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)    # 对数据进行标准化
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX-meanTrain) / varTrain
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)    # 测试回归效果并存储
            errorMat[i, k] = ((yEst.T.A - np.array(testY)) ** 2).sum()    # 计算误差
    meanErrors = np.mean(errorMat, 0)    # 计算误差估计值的均值
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    # 输出构建的模型
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))

def regression1():
    xArr, yArr = loadDataSet(r"D:\Data\ML_InAction\Regression\data.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws = standRegres(xArr, yArr)    # 估计模型的未知系数的值
    fig = plt.figure()
    ax = fig.add_subplot(111)    # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])     # flatten()函数将数组或者矩阵一维化
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws    # 拟合结果值
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

def regression2():
    xArr, yArr = loadDataSet(r"D:\Data\ML_InAction\Regression\data.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter([xMat[:, 1].flatten().A[0]], [np.mat(yArr).T.flatten().A[0]], s=2, c='red')
    plt.show()

def abaloneTest():
    abX, abY = loadDataSet(r'D:\Data\ML_InAction\Regression\abalone.txt')    # 加载数据
    abY = np.array(abY)
    # 使用不同的核进行预测
    oldyHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    oldyHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    oldyHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", rssError(abY[0:99], oldyHat01.T))
    print("old yHat1 error Size is :", rssError(abY[0:99], oldyHat1.T))
    print("old yHat10 error Size is :", rssError(abY[0:99], oldyHat10.T))
    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("new yHat01 error Size is :", rssError(abY[0:99], newyHat01.T))
    newyHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("new yHat1 error Size is :", rssError(abY[0:99], newyHat1.T))
    newyHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("new yHat10 error Size is :", rssError(abY[0:99], newyHat10.T))
    # 使用简单的线性回归进行预测，与上面的计算进行比较
    standWs = standRegres(abX[0:99], abY[0:99])
    standyHat = np.mat(abX[100:199]) * standWs
    print("standRegress error Size is:", rssError(abY[100:199], standyHat.T.A))

def regression3():
    abX, abY = loadDataSet(r'D:\Data\ML_InAction\Regression\abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def regression4():
    xArr, yArr = loadDataSet(r'D:\Data\ML_InAction\Regression\abalone.txt')
    stageWise(xArr, yArr, 0.01, 200)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = regularize(xMat)
    yM = np.mean(yMat, 0)
    yMat = yMat - yM
    weights = standRegres(xMat, yMat.T)
    print(weights.T)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(weights.T)
    # plt.show()

def regression5():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY, 10)

if __name__ == '__main__':
    # regression1()    # 简单回归函数
    regression2()     # 局部加权回归函数
    # abaloneTest()   # 简单回归函数
    # regression3()   # 岭回归函数
    # regression4()   # 逐步向前回归函数
    # regression5()
    #pass