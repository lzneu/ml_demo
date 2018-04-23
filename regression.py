#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : regression.py
# @Author: 投笔从容
# @Date  : 2018/4/23
# @Desc  : 手写线性回归

from numpy import *


def loadDataSet(filename):
    path = './data/'
    numFeat = len(open(path + filename).readline().split('\t')) -1
    dataMat = []
    labelMat = []
    fr = open(path + filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat

    # 若（XTX）不可逆
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# 局部加权线性回归（LOCALLY WEIGHTED LINEAR REGRESSION） LWLR
# 增加了计算量 因为他对每个点做预测时都必须使用整个数据集
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * weights * xMat
    # 若（XTX）不可逆
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * weights * yMat)
    return testPoint * ws

if __name__ == '__main__':

    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    # 绘制图像和拟合直线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # 将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    Y = yHat.T
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


