#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : adaboost.py
# @Author: 投笔从容
# @Date  : 2018/5/6
# @Desc  : 手写实现adaboost

from numpy import *


def loadSimpData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLaels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLaels


def stumpClassfy(dataMat, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类
    所有在阈值一边的数据会分到类别-1 而在另一边的数据会分到类别+1 该函数可以通过数组过滤来实现首先返回数组的全部元素置1 然后将所有不满足不等式要求的元素设置为-1
    :return:
    '''
    dataMat = mat(dataMat)
    retArray = ones((shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    第一层for循环在数据集上的所有特征遍历， 第二层for循环在在这些值上遍历 第三层是在大于和小于之间切换
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    numsteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minErr = inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numsteps
        for j in range(-1, int(numsteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + stepSize * float(j))
                predictVal =stumpClassfy(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictVal == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim: %d, thresh: %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))

                if weightedError < minErr:
                    bestClasEst = predictVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    minErr = weightedError
    return bestStump, minErr, bestClasEst


# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClaasEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print('D: ', D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 1e-16避免错误时除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClaasEst += alpha * classEst
        # print('aggClassEst: ', aggClaasEst)
        errCounts = sign(aggClaasEst) != mat(classLabels).T
        aggErrors = multiply(errCounts, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total Error: ', errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr


# Adaboost分类函数
def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassfy(dataToClass, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classEst * classifierArr[i]['alpha']
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))  #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 绘制ROC曲线以及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0





if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('data/horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('data/horseColicTest2.txt')
    weakClassArr = adaBoostTrainDS(dataArr, labelArr, 80)
    prediction10 = adaClassify(testArr, weakClassArr)
    errArr = mat(ones((67, 1)))
    errRate = (errArr[prediction10 != mat(testLabelArr).T].sum()) / 67.0
    print(errRate)
