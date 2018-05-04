#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : SVM.py
# @Author: 投笔从容
# @Date  : 2018/5/4
# @Desc  : 手写 SVM ,SMO算法实现简化版本 SMO算法完整版
import random
from numpy import *

# 构建两个辅助函数
#   1 用于在某个区间范围内随机选择一个整数
#   2 用于数值太大时对其进行调整
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 剪辑
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


# 简化版本SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数
    :param toler: 容错率
    :param maxIter: 最大循环次数
    :return: b alphas
    '''
    # 初始化
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMat)
    alphas = mat(zeros((m, 1)))
    iter = 0  # 改变两存储的则是在没有任何alpha改变的情况下遍历数据集的次数
    while (iter < maxIter):
        alphaPairsChanged = 0  # 记录alpha是否已经优化
        for i in range(m):  # 外循环
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 如果alpha 可以更改进入优化过程
            if ((labelMat[i]*Ei < - toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 内循环
                fXj = float(multiply(alphas, labelMat).T * (dataMat*dataMat[j, :].T)) + b
                Ej = fXj - float(classLabels[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print('L==H')
                    continue
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :]*dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                # 迭代求最优解
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 剪辑 alpha2
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 判断是否停
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                # alpha1
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])

                # 更新b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * (dataMat[i, :] * dataMat[i, :].T) - labelMat[j] * (alphas[j] - alphaJold) * (dataMat[j, :] * dataMat[i, :].T)
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * (dataMat[i, :] * dataMat[2, :].T) - labelMat[j] * (alphas[j] - alphaJold) * (dataMat[j, :] * dataMat[j, :].T)
                #  这个地方有写不理解
                if (0 < alphas[i]) and (C > alphas[i]):  # 0 < alpha1 < C
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):  # 0 < alpha2 < C
                    b = b2
                else:   # alpha1 alpha2 是 0 或者 C
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print('iter: %d i: %d ,pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            print('iteration number: %d' % iter)

    return b, alphas


class optStruct:
    def __init__(self, dataMat, classLabels, C, toler, kTup):
        self.X = dataMat
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMat)[0]
        self.alphas = mat(zeros((self.m ,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 误差缓存 第一列为是否有效的标志 第二列为实际的E值
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

# 计算误差函数
def calEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.K[:, k])) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 内循环的启发式方法
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # Return the indices of the elements that are non-zero.
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek

        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 完整的platt SMO算法中的优化例程
def innerL(i, oS):
    Ei = calEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C - oS.alphas[i] + oS.alphas[j])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:
            print('L == H')  # 没有符合条件的样本
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        # 更新b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.K[i, i]) * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * (oS.K[j, i]) * (oS.alphas[j] - alphaJold)
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.K[i, j]) * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * (oS.K[j, j]) * (oS.alphas[j] - alphaJold)
        if (oS.alphas[i] < oS.C) and (oS.alphas[i] > 0):
            oS.b = b1
        elif (oS.alphas[j] < oS.C) and (oS.alphas[j] > 0):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 完整版Platt SMO的外循环代码
def smoP(dataMat, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMat), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 当迭代次数超过max 或者 遍历整个集合都未对任意alpha对进行修改 就退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 遍历整个数据集
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 一个innerL优化一对
                print('full dataSet, iter: %d, i: %d, pairs changed: %d' % (iter, i, alphaPairsChanged))
            iter += 1
        # 遍历非边界值
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # nonzero 返回 (非零元素的行index list, 非零元素的列index list)
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter:iter: %d, i: %d, pairs changed: %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False

        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas


# 计算w
def calWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def kernelTrans(X, A, kTup):
    '''
    核转换函数
    :param X: 数据集
    :param A: 第i个样本
    :param kTup: 和函数选择参数
    :return: 计算后的核函数内积值向量
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('Houston We have a problem, that kernel is not recognized')
    return K


# 测试利用核技巧的svm
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('data/svm/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('data/svm/testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))



if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet('data/svm/testSet.txt')
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    # print(alphas)
    # # 了解哪些点是支持向量
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(dataArr[i], labelArr[i])


    # dataArr, labelArr = loadDataSet('data/svm/testSet.txt')
    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # ws = calWs(alphas, dataArr, labelArr)
    # print(ws)
    # dataMat = mat(dataArr)
    # fx0 = dataMat[0] * mat(ws) + b
    # print(fx0)

    testRbf()