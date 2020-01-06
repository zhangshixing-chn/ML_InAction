#********========AdaBoost：sklearn算法========********#
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import metrics
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
# # 创建数据集及模型
# rng = np.random.RandomState(1)
# X = np.linspace(0, 6, 100)[:, np.newaxis]    # 将数组X转化为Nx1的矩阵格式
# y = np.sin(X).ravel() + np.sin(6 * X).ravel()+ rng.normal(0, 0.1, X.shape[0])   # ravel()函数，将多为矩阵转化为1维数组
# regr_1 = DecisionTreeRegressor(max_depth = 4)
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 1000, random_state = rng)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# # 预测模型及绘制结果
# y_1 = regr_1.predict(X)
# y_2 = regr_2.predict(X)
# plt.figure()
# plt.scatter(X, y, c="k", label="training samples")
# plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
# plt.plot(X, y_2, c="r", label="n_estimators=1000", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Boosted Decision Tree Regression")
# plt.legend()
# plt.show()
# print('y---', type(y[0]), len(y), y[:4])
# print('y_1---', type(y_1[0]), len(y_1), y_1[:4])
# print('y_2---', type(y_2[0]), len(y_2), y_2[:4])
# # 适合2分类
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# print('y_scores---', type(y_scores[0]), len(y_scores), y_scores)
# print(metrics.roc_auc_score(y_true, y_scores))

#********========AdaBoost：算法案例========********#
def load_sim_data():
	'''
	:return
		data_arr: feature对应的数据集
		label_arr: feature对应的分类标签
	'''
	data_mat = np.mat([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
	class_label = [1.0, 1.0, -1.0, -1.0, 1.0]
	return data_mat, class_label

def load_data_set(filename):   # 对于存在多列数据的读取方式
	num_feat = len(open(filename).readline().split('\t'))    # 数据包含的列数
	data_arr = []
	label_arr = []
	fr = open(filename)
	for line in fr.readlines():
		line_arr = []
		cur_line = line.strip().split('\t')   # 对数据的某一行进行切割，得到每行的数组
		for i in range(num_feat - 1):
			line_arr.append(float(cur_line[i]))   # 将每一行切割后的每一列值添加到行列表中
		data_arr.append(line_arr)
		label_arr.append(float(cur_line[-1]))
	return np.mat(data_arr), label_arr

def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):     #  将数据集按照feature列的value，进行二分法切分，比较阈值来赋值分类
	'''
	:param data_mat: Matrix数据集
	:param dimen: 特征的哪一个列
	:param thresh_val: 特征列要比较的值
	:param thresh_ineq:
	:return: np.array
	'''
	ret_array = np.ones((np.shape(data_mat)[0], 1))
	if thresh_ineq == 'lt':     # lt表示修改左边的值，gt表示修改右边的值
		ret_array[data_mat[:, dimen] <= thresh_val] = -1.0     # data_mat[:, dimen]表示数据集中第dimen列的所有值
	else:
		ret_array[data_mat[:, dimen] > thresh_val] = -1.0
	return ret_array

def build_stump(data_arr, class_labels, D):   # 得到决策树的模型  D表示最初的特征权重值
	'''
	:param data_arr: 特征数据集合
	:param class_labels: 标签集合
	:param D: 初始化的特征权重值
	:return:
		bestStump: 最优分类器模型
		min_error: 最小错误率
		best_class_est: 训练后的结果集
	'''
	data_mat = np.mat(data_arr)
	label_mat = np.mat(class_labels).T
	m, n = np.shape(data_mat)
	num_steps = 10.0
	best_stump = {}
	best_class_est = np.mat(np.zeros((m, 1)))
	min_err = np.inf   # 无穷大
	for i in range(n):
		range_min = data_mat[:, i].min()
		range_max = data_mat[:, i].max()
		step_size = (range_max - range_min) / num_steps
		for j in range(-1, int(num_steps) + 1):
			for inequal in ['It', 'gt']:
				thresh_val = (range_min + float(j) * step_size)
				predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
				err_arr = np.mat(np.ones((m, 1)))
				err_arr[predicted_vals == label_mat] = 0
				weighted_err = D.T * err_arr    # 这里是矩阵乘法
				'''
				dim            表示 feature列
				thresh_val      表示树的分界值
				inequal        表示计算树左右颠倒的错误率的情况
				weighted_error  表示整体结果的错误率
				best_class_est    预测的最优结果 （与class_labels对应）
				'''
				if weighted_err < min_err:
					min_err = weighted_err
					best_class_est = predicted_vals.copy()
					best_stump['dim'] = i
					best_stump['thresh'] = thresh_val
					best_stump['ineq'] = inequal
	return best_stump, min_err, best_class_est   # best_stump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少 (单个弱分类器)

def ada_boost_train_ds(data_arr, class_labels, num_it = 40):   # ada_boost训练过程
	'''
	:param data_arr: 特征标签集合
	:param class_labels: 分类标签集合
	:param num_it: 迭代次数
	:return:
		week_class_arr: 弱分类器的集合
		agg_class_est: 预测的分类结果值
	'''
	weak_class_arr = []
	m = np.shape(data_arr)[0]
	D = np.mat(np.ones((m, 1)) / m)    # 初始化 D，设置每个特征的权重值，平均分为m份
	agg_class_est = np.mat(np.zeros((m, 1)))
	for i in range(num_it):
		best_stump, error, class_est = build_stump(data_arr, class_labels, D)    # 得到决策树模型
		#print('D: {}'.format(D.T))
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))   # alpha 目的主要是计算每一个分类器实例的权重(加和就是分类结果)
		best_stump['alpha'] = alpha   # 计算每个分类器的 alpha 权重值
		weak_class_arr.append(best_stump)
		#print('class_est: {}'.format(class_est.T))
		# 分类正确：乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
		# 分类错误：乘积为 -1，结果会受影响，所以也乘以 -1
		expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
		#print('(-1取反)预测值 expon=', expon.T)
		# 判断正确的，就乘以-1，否则就乘以1， 为什么？ 书上的公式。
		# print('(-1取反)预测值 expon=', expon.T)
		# 计算e的expon次方，然后计算得到一个综合的概率的值
		# 结果发现： 判断错误的样本，D对于的样本权重值会变大。
		# multiply是对应项相乘
		D = np.multiply(D, np.exp(expon))
		D = D / D.sum()
		# 预测的分类结果值，在上一轮结果的基础上，进行加和操作
		#print('叠加前的分类结果class_est: {}'.format(class_est.T))
		agg_class_est += alpha * class_est
		#print('叠加后的分类结果agg_class_est: {}'.format(agg_class_est.T))
		# sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
		# 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
		agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
		error_rate = agg_errors.sum() / m
		#print('total error: {}\n'.format(error_rate))
		if error_rate == 0.0:
			break
	return weak_class_arr, agg_class_est

def ada_classify(data_to_class, classifier_arr):    # 弱分类器的集合进行预测
	'''
	:param data_to_class: 数据集
	:param classifier_arr: 分类器列表
	:return: 分类结果：+1 或者 -1
	'''
	data_mat = np.mat(data_to_class)
	m = np.shape(data_mat)[0]
	agg_class_est = np.mat(np.zeros((m, 1)))
	for i in range(len(classifier_arr)):
		class_est = stump_classify(data_mat, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
		agg_class_est += classifier_arr[i]['alpha'] * class_est
		#print(agg_class_est)
	return np.sign(agg_class_est)

def plot_roc(pred_strengths, class_labels):
	'''
	:param pred_strengths: 最终预测结果的权重值
	:param class_labels: 原始数据的分类结果集
	:return:
	'''
	y_sum = 0.0
	num_pos_class = np.sum(np.array(class_labels) == 1.0)   # 正样本求和
	y_step = 1 / float(num_pos_class)   # 正样本概率
	x_step = 1 / float(len(class_labels) - num_pos_class)    # 负样本概率
	sorted_indicies = pred_strengths.argsort()   # np.argsort函数返回的是数组值从小到大的索引值
	fig = plt.figure()
	fig.clf()
	#ax = plt.subplot(111)
	cur = (1.0, 1.0)   # 光标值
	for index in sorted_indicies.tolist()[0]:
		#print(index)     # 0
		if class_labels[index] == 1.0:
			del_x = 0
			del_y = y_step
		else:
			del_x = x_step
			del_y = 0
			y_sum += cur[1]
		# 画点连线 (x1, x2, y1, y2)
		print(cur[0], cur[0] - del_x, cur[1], cur[1] - del_y)
		plt.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], 'b-*')
		cur = (cur[0] - del_x, cur[1] - del_y)
	# 画对角的虚线线
	plt.plot([0, 1], [0, 1], 'b--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve for AdaBoost horse colic detection system')
	plt.axis([0, 1, 0, 1])   # 设置画图的范围区间 (x1, x2, y1, y2)
	plt.show()
	#print("the Area Under the Curve is: ", y_sum * x_step)

def test():
	data_mat, class_labels = load_data_set(r'D:\Data\ML_InAction\集成算法\horse_ColicTraining2.txt')
	#print(data_mat.shape, len(class_labels))
	weak_class_arr, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 40)
	#print(weak_class_arr, '\n-----\n', agg_class_est.T)
	plot_roc(agg_class_est, class_labels)
	data_arr_test, label_arr_test = load_data_set(r'D:\Data\ML_InAction\集成算法\horse_ColicTest2.txt')
	m = np.shape(data_arr_test)[0]
	predicting10 = ada_classify(data_arr_test, weak_class_arr)
	err_arr = np.mat(np.ones((m, 1)))
	# 测试：计算总样本数，错误样本数，错误率
	print(m, err_arr[predicting10 != np.mat(label_arr_test).T].sum(), err_arr[predicting10 != np.mat(label_arr_test).T].sum() / m)

if __name__ == '__main__':
	test()