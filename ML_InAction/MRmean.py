# #********========MapReduce：均值实现========********#
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRmean(MRJob):
    def __init__(self, *args, **kwargs):  # 对数据初始化
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    # 接受输入数据流
    def map(self, key, val):  # 需要 2 个参数，求数据的和与平方和
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal*inVal

    # 所有输入到达后开始处理
    def map_final(self):  # 计算数据的平均值，平方的均值，并返回
        if self.inCount == 0:
            return
        mn = self.inSum/self.inCount
        mnSq = self.inSqSum/self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumN, cumVal, cumSumSq = 0.0, 0.0, 0.0
        for valArr in packedValues:  # 从输入流中获取值
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj*float(valArr[1])
            cumSumSq += nj*float(valArr[2])
        mean = cumVal/cumN
        var = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
        yield (mean, var)  # 发出平均值和方差

    def steps(self):
        """
        step方法定义执行的步骤。
        执行顺序不必完全遵循map-reduce模式。
        例如：
            1. map-reduce-reduce-reduce
            2. map-reduce-map-reduce-map-reduce
        在step方法里，需要为mrjob指定mapper和reducer的名称。如果没有，它将默认调用mapper和reducer方法。
        在mapper 和 mapper_final中还可以共享状态，mapper 或 mapper_final 不能 reducer之间共享状态。
        """
        return [MRStep(mapper=self.map, mapper_final=self.map_final, reducer=self.reduce)]

if __name__ == '__main__':
    MRmean.run()

# #********========MapReduce：均值实现（Mapper阶段）========********#
import sys
from numpy import mat, mean, power

'''
    这个mapper文件按行读取所有的输入并创建一组对应的浮点数，然后得到数组的长度并创建NumPy矩阵。
    再对所有的值进行平方，最后将均值和平方后的均值发送出去。这些值将用来计算全局的均值和方差。
    Args：
        file 输入数据
    Return：
'''

def read_input(file):
    for line in file:
        yield line.rstrip()             # 返回一个 yield 迭代器，每次获取下一个值，节约内存。

input = read_input(sys.stdin)            # 创建一个输入的数据行的列表list
input = [float(line) for line in input]  # 将得到的数据转化为 float 类型
numInputs = len(input)                   # 获取数据的个数，即输入文件的数据的行数
input = mat(input)                       # 将 List 转换为矩阵
sqInput = power(input, 2)                # 将矩阵的数据分别求 平方，即 2次方

# 输出 数据的个数，n个数据的均值，n个数据平方之后的均值
# 第一行是标准输出，也就是reducer的输出
# 第二行识标准错误输出，即对主节点作出的响应报告，表明本节点工作正常。
# 【这不就是面试的装逼重点吗？如何设计监听架构细节】注意：一个好的习惯是想标准错误输出发送报告。如果某任务10分钟内没有报告输出，则将被Hadoop中止。
print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))  # 计算均值
print("map report: still alive", file=sys.stderr)

#********========MapReduce：均值实现（Reduce阶段）========********#
import sys

'''
    mapper 接受原始的输入并产生中间值传递给 reducer。
    很多的mapper是并行执行的，所以需要将这些mapper的输出合并成一个值。
    即：将中间的 key/value 对进行组合。
'''

def read_input(file):
    for line in file:
        yield line.rstrip()						# 返回值中包含输入文件的每一行的数据的一个大的List
input = read_input(sys.stdin)					# 创建一个输入的数据行的列表list

# 将输入行分割成单独的项目并存储在列表的列表中
mapperOut = [line.split('\t') for line in input]
# 输入 数据的个数，n个数据的均值，n个数据平方之后的均值
print (mapperOut)

# 累计样本总和，总和 和 平分和的总和
cumN, cumVal, cumSumSq = 0.0, 0.0, 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
# 计算均值( varSum是计算方差的展开形式 )
mean_ = cumVal/cumN
varSum = (cumSumSq - 2*mean_*cumVal + cumN*mean_*mean_)/cumN
# 输出 数据总量，均值，平方的均值（方差）
print("数据总量：%d\t均值：%f\t方差：%f" % (cumN, mean_, varSum))
print("reduce report: still alive", file=sys.stderr)

#********========MapReduce：SVM实现========********#
import pickle
from numpy import *
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRsvm(MRJob):
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def __init__(self, *args, **kwargs):
        super(MRsvm, self).__init__(*args, **kwargs)
        self.data = pickle.load(open('/opt/git/MachineLearnidata/15.BigData_MapReduce/svmDat27', 'r'))
        self.w = 0
        self.eta = 0.69
        self.dataList = []
        self.k = self.options.batchsize
        self.numMappers = 1
        self.t = 1  # iteration number

    def configure_args(self):
        super(MRsvm, self).configure_args()
        self.add_passthru_arg(
            '--iterations', dest='iterations', default=2, type=int,
            help='T: number of iterations to run')
        self.add_passthru_arg(
            '--batchsize', dest='batchsize', default=100, type=int,
            help='k: number of data points in a batch')

    def map(self, mapperId, inVals):  # 需要 2 个参数
        # input: nodeId, ('w', w-vector) OR nodeId, ('x', int)
        if False:
            yield
        if inVals[0] == 'w':                  # 积累 w向量
            self.w = inVals[1]
        elif inVals[0] == 'x':
            self.dataList.append(inVals[1])   # 累积数据点计算
        elif inVals[0] == 't':                # 迭代次数
            self.t = inVals[1]
        else:
            self.eta = inVals                 # 这用于 debug， eta未在map中使用

    def map_fin(self):
        labels = self.data[:, -1]
        X = self.data[:, :-1]                # 将数据重新形成 X 和 Y
        if self.w == 0:
            self.w = [0.001] * shape(X)[1]   # 在第一次迭代时，初始化 w
        for index in self.dataList:
            p = mat(self.w)*X[index, :].T    # calc p=w*dataSet[key].T
            if labels[index]*p < 1.0:
                yield (1, ['u', index])      # 确保一切数据包含相同的key
        yield (1, ['w', self.w])             # 它们将在同一个 reducer
        yield (1, ['t', self.t])

    def reduce(self, _, packedVals):
        for valArr in packedVals:            # 从流输入获取值
            if valArr[0] == 'u':
                self.dataList.append(valArr[1])
            elif valArr[0] == 'w':
                self.w = valArr[1]
            elif valArr[0] == 't':
                self.t = valArr[1]

        labels = self.data[:, -1]
        X = self.data[:, 0:-1]
        wMat = mat(self.w)
        wDelta = mat(zeros(len(self.w)))

        for index in self.dataList:
            wDelta += float(labels[index]) * X[index, :]  # wDelta += label*dataSet
        eta = 1.0/(2.0*self.t)       # calc new: eta
        # calc new: w = (1.0 - 1/t)*w + (eta/k)*wDelta
        wMat = (1.0 - 1.0/self.t)*wMat + (eta/self.k)*wDelta
        for mapperNum in range(1, self.numMappers+1):
            yield (mapperNum, ['w', wMat.tolist()[0]])    # 发出 w
            if self.t < self.options.iterations:
                yield (mapperNum, ['t', self.t+1])        # 增量 T
                for j in range(self.k/self.numMappers):   # emit random ints for mappers iid
                    yield (mapperNum, ['x', random.randint(shape(self.data)[0])])

    def steps(self):
        return [MRStep(mapper=self.map, reducer=self.reduce, mapper_final=self.map_fin)] * self.options.iterations

from mrjob.protocol import JSONProtocol
from numpy import *

fw=open('kickStart2.txt', 'w')
for i in [1]:
    for j in range(100):
        fw.write('["x", %d]\n' % random.randint(200))
fw.close()

if __name__ == '__main__':
    MRsvm.run()

#********========MapReduce：pegasos实现========********#
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        # dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def seqPegasos(dataSet, labels, lam, T):
    m, n = shape(dataSet)
    w = zeros(n)
    for t in range(1, T+1):
        i = random.randint(m)
        eta = 1.0/(lam*t)
        p = predict(w, dataSet[i, :])
        if labels[i]*p < 1:
            w = (1.0 - 1/t)*w + eta*labels[i]*dataSet[i, :]
        else:
            w = (1.0 - 1/t)*w
        print(w)
    return w


def predict(w, x):
    return w*x.T  # 就是预测 y 的值


def batchPegasos(dataSet, labels, lam, T, k):
    """batchPegasos()
    Args:
        dataMat    特征集合
        labels     分类结果集合
        lam        固定值
        T          迭代次数
        k          待处理列表大小
    Returns:
        w          回归系数
    """
    m, n = shape(dataSet)
    w = zeros(n)  # 回归系数
    dataIndex = list(range(m))
    for t in range(1, T+1):
        wDelta = mat(zeros(n))  # 重置 wDelta

        # 它是学习率，代表了权重调整幅度的大小。（也可以理解为随机梯度的步长，使它不断减小，便于拟合）
        # 输入T和K分别设定了迭代次数和待处理列表的大小。在T次迭代过程中，每次需要重新计算eta
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):      # 全部的训练集  内循环中执行批处理，将分类错误的值全部做累加后更新权重向量
            i = dataIndex[j]
            p = predict(w, dataSet[i, :])              # mapper 代码

            # 如果预测正确，并且预测结果的绝对值>=1，因为最大间隔为1, 认为没问题。
            # 否则算是预测错误, 通过预测错误的结果，来累计更新w.
            if labels[i]*p < 1:                        # mapper 代码
                wDelta += labels[i]*dataSet[i, :].A    # 累积变化
        # w通过不断的随机梯度的方式来优化
        w = (1.0 - 1/t)*w + (eta/k)*wDelta             # 在每个 T上应用更改
        # print '-----', w
    # print '++++++', w
    return w

datArr, labelList = loadDataSet('data/15.BigData_MapReduce/testSet.txt')
datMat = mat(datArr)
# finalWs = seqPegasos(datMat, labelList, 2, 5000)
finalWs = batchPegasos(datMat, labelList, 2, 50, 100)
print(finalWs)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
x1 = []
y1 = []
xm1 = []
ym1 = []
for i in range(len(labelList)):
    if labelList[i] == 1.0:
        x1.append(datMat[i, 0])
        y1.append(datMat[i, 1])
    else:
        xm1.append(datMat[i, 0])
        ym1.append(datMat[i, 1])
ax.scatter(x1, y1, marker='s', s=90)
ax.scatter(xm1, ym1, marker='o', s=50, c='red')
x = arange(-6.0, 8.0, 0.1)
y = (-finalWs[0, 0]*x - 0)/finalWs[0, 1]
# y2 = (0.43799*x)/0.12316
y2 = (0.498442*x)/0.092387  # 2 iterations
ax.plot(x, y)
ax.plot(x, y2, 'g-.')
ax.axis([-6, 8, -4, 5])
ax.legend(('50 Iterations', '2 Iterations'))
plt.show()

#********========MapReduce：proximalSVM实现========********#
import base64
import pickle
import numpy

def map(key, value):
   # input key= class for one training example, e.g. "-1.0"
   classes = [float(item) for item in key.split(",")]   # e.g. [-1.0]
   D = numpy.diag(classes)
   # input value = feature vector for one training example, e.g. "3.0, 7.0, 2.0"
   featurematrix = [float(item) for item in value.split(",")]
   A = numpy.mat(featurematrix)
   # create matrix E and vector e
   e = numpy.mat(numpy.ones(len(A)).reshape(len(A), 1))
   E = numpy.mat(numpy.append(A, -e, axis=1))
   # create a tuple with the values to be used by reducer
   # and encode it with base64 to avoid potential trouble with '\t' and '\n' used
   # as default separators in Hadoop Streaming
   producedvalue = base64.b64encode(pickle.dumps((E.T*E, E.T*D*e)))
   # note: a single constant key "producedkey" sends to only one reducer
   # somewhat "atypical" due to low degree of parallism on reducer side
   print("producedkey\t%s" % (producedvalue))

def reduce(key, values, mu=0.1):
  sumETE = None
  sumETDe = None

  # key isn't used, so ignoring it with _ (underscore).
  for _, value in values:
    # unpickle values
    ETE, ETDe = pickle.loads(base64.b64decode(value))
    if sumETE == None:
      # create the I/mu with correct dimensions
      sumETE = numpy.matrix(numpy.eye(ETE.shape[1])/mu)
    sumETE += ETE

    if sumETDe == None:
      # create sumETDe with correct dimensions
      sumETDe = ETDe
    else:
      sumETDe += ETDe

    # note: omega = result[:-1] and gamma = result[-1]
    # but printing entire vector as output
    result = sumETE.I*sumETDe
    print("%s\t%s" % (key, str(result.tolist())))

#********========MapReduce：MRWordCountUtility实现========********#
from mrjob.job import MRJob

class MRWordCountUtility(MRJob):

    def __init__(self, *args, **kwargs):
        super(MRWordCountUtility, self).__init__(*args, **kwargs)
        self.chars = 0
        self.words = 0
        self.lines = 0

    def mapper(self, _, line):
        if False:
            yield  # I'm a generator!

        self.chars += len(line) + 1  # +1 for newline
        self.words += sum(1 for word in line.split() if word.strip())
        self.lines += 1

    def mapper_final(self):
        yield('chars', self.chars)
        yield('words', self.words)
        yield('lines', self.lines)

    def reducer(self, key, values):
        yield(key, sum(values))

if __name__ == '__main__':
    MRWordCountUtility.run()