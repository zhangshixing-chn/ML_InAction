#********=========DescionTree: sklearn库中的算法========********#
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def createDataSet():
	data = []
	labels = []
	with open(r'D:\Data\ML_InAction\DescionTree\data.txt') as ifile:
		for line in ifile:
			tokens = line.strip().split(' ')
			data.append([float(tk) for tk in tokens[:-1]])
			labels.append(tokens[-1])
	x = np.array(data)    # 特征数据
	labels = np.array(labels)    # label分类的标签数据
	y = np.zeros(labels.shape)     # 预估结果的标签数据
	y[labels == 'fat'] = 1      # 转换为0/1
	print(data, '-------', x, '-------', labels, '-------', y)
	return x, y

def predict_train(x_train, y_train):
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	print(clf)
	clf.fit(x_train, y_train)
	print('feature_importances_: %s' % clf.feature_importances_)
	y_pre = clf.predict(x_train)
	print(x_train)
	print(y_pre)
	print(y_train)
	print(np.mean(y_pre == y_train))
	return y_pre, clf

def show_precision_recall(x, y, clf, y_train, y_pre):
	precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
	answer = clf.predict_proba(x)[:, 1]      # 计算全量的预估结果
	target_names = ['thin', 'fat']      # target_name是以y的label为分类标准
	print(classification_report(y, answer, target_names=target_names))
	print(answer)
	print(y)

def show_pdf(clf):    # 可视化输出
	import os
	#import graphviz
	import pydotplus
	from six import StringIO
	#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file=dot_data)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(r"D:\Data\ML_InAction\DescionTree\tree.pdf")

if __name__ == '__main__':
	x, y = createDataSet()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	print('拆分数据：', x_train, x_test, y_train, y_test)
	y_pre, clf = predict_train(x_train, y_train)
	show_precision_recall(x, y, clf, y_train, y_pre)
	show_pdf(clf)