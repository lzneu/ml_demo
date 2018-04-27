#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : LR.py
# @Author: 投笔从容
# @Date  : 2018/4/27
# @Desc  : 手写 Logistic回归
from math import exp
from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data/testSet.txt')
    for line in fr.readlines():
        line = line.strip().split()
        dataMat.append([1.0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat, labelMat


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


# 梯度提升算法
def gradAscend(dataMatIn, classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    alpha = 0.001
    maxCycle = 500
    weights = ones((n, 1))
    for k in range(maxCycle):
        # 预测
        h = sigmoid(dataMat * weights)
        errors = labelMat - h
        weights = weights + alpha * (dataMat.transpose() * errors)
    return weights


# 画出数据集 和lr最佳拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataMat)[0]
    # 记录两种分类点
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='r', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

'''
随机梯度上升算法中的变量均为numpy数组 ndarray
'''
# 随机梯度提升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLaebl, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = list(range(m))   # 用来随机样本选取顺序
        for i in range(m):
            # 每次迭代时需要调整alpha 精度提升
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLaebl[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# 用罗吉斯特回归预测马疝病的死亡率
def classifyVector(intX, weights):
    prob = sigmoid(sum(intX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open('./data/horseColicTraining.txt')
    frTest = open('./data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        linArr = []
        for i in range(21):
            linArr.append(float(currLine[i]))
        trainingSet.append(linArr)
        trainingLabels.append(float(currLine[21]))

    trainingWeights = stocGradAscent1(array(trainingSet), array(trainingLabels), 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classifyVector(array(lineArr), trainingWeights) != float(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('the error rate of this test is : %f' % errorRate)\

    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('After %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))



if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet()
    # weights = stocGradAscent1(array(dataArr), array(labelArr), 500)
    # plotBestFit(array(weights))
    multiTest()





