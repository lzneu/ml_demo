#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : kNN.py
# @Author: 投笔从容
# @Date  : 2018/4/20
# @Desc  : k-近邻 算法练习

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVectors = int(m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVectors):
        classFierResult = classify0(normDataSet[i, :], normDataSet[numTestVectors:m, :], datingLabels[numTestVectors: m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classFierResult, datingLabels[i]))
        if classFierResult != datingLabels[i]:
            errorCount += 1.0
    print('the total error rate is : %f' % (errorCount / numTestVectors))


# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 使用tile函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def file2matrix(filename):
    # 获得文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOlines = len(arrayOLines)

    # 创建返回的numpy矩阵
    returnMat = zeros((numberOlines, 3))
    classLabelVector = []
    index = 0
    # 解析文件到数据
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(intX, dataSet, labels, k):
    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDisstances = sqDiffMat.sum(axis=1)
    distances = sqDisstances ** 0.5

    # 选择k个距离最小的点
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 将一个32* 32的图像转化为一个1*1024的向量
def img2vector(filename):
    returnVector = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVector[0, 32*i+j] = int(lineStr[j])

    return returnVector

# 测试使用k近邻算法识别手写数字
def handwritingClassTest():

    # 构建训练集
    hwLabels = []
    trainingFileList = listdir('./data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(int(classNumStr))
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)
    # 构建测试集并进行测试
    testFileList = listdir('./data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        testLabel = int(classNumStr)
        vectorUnderTest = img2vector('./data/testDigits/%s' % fileNameStr)
        classLabel = classify0(vectorUnderTest, trainingMat, hwLabels, 5)
        print('the classfier came back with: %d, the real answer is : %d' % (classLabel, testLabel))
        if testLabel != classLabel:
            errorCount += 1.0
    print('the error rate is : %f' % (errorCount/float(mTest)))



if __name__ == '__main__':
    '''绘图看数据'''
    # datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # # print(normDataSet)
    # # print(ranges)
    # # print(minVals)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # plt.show()
    '''约会定位'''
    # datingClassTest()
    '''识别手写体数字'''
    # testVector = img2vector('./data/trainingDigits/0_0.txt')
    # print(testVector)
    handwritingClassTest()
