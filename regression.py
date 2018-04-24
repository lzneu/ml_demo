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

# 测试线性回归
def testStandRegres():
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


# 测试Lwlr
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# 定义均方误差
def rssError(yArr, yHar):
    return ((yArr - yHar) ** 2).sum()


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) *lam
    if linalg.det(denom) == 0.0:
        print('This matrix is singular ,cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# 测试岭回归
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 归一化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):
    xmeans = mean(xMat, 0)
    xvars = var(xMat, 0)

    return (xMat-xmeans) / xvars

# 向前逐步线性回归, 为一种贪心算法
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    ymat = mat(yArr).T
    # 归一化
    xMat = regularize(xMat)
    ymat = ymat - mean(ymat, 0)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    # 迭代
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += sign*eps
                yTest = xMat * wsTest
                rssE = rssError(ymat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

from time import sleep
import json
import urllib
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(1)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


if __name__ == '__main__':

    # # 测试局部加权线性回归
    # xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr, xArr, yArr, 0.03)
    # # 对xArr进行排序
    # xMat = mat(xArr)
    # yMat = mat(yArr )
    # srtInd = xMat[:, 1].argsort(0)
    # xSort = xMat[srtInd][:, 0, :]
    #
    # # 画出曲线
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # ax.plot(xSort[:, 1], yHat[srtInd])
    # plt.show()


    # # 预测鲍鱼年龄
    # abx, aby = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abx, aby)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot([(i-10) for i in range(30)], ridgeWeights)
    # plt.show()

    # # 向前逐步回归
    # xArr, yArr = loadDataSet('abalone.txt')
    # weightsMat = stageWise(xArr, yArr, 0.001, 5000)

    # 预测乐高玩具套装价格
    logX = []
    logY = []
    setDataCollect(logX, logY)

    print('done!')



