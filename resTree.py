#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : getTree.py
# @Author: 投笔从容
# @Date  : 2018/4/24
# @Desc  : 手写回归树 模型树
from numpy import *


# 载入数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltLine = list(map(float, curline))
        dataMat.append(fltLine)
    return dataMat


# 数据切分函数
def binSplitDataSet(dataSet, feature, value):

    mat0 = dataSet[nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1


# 创建叶子结点的函数
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 总方差计算函数
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

# 模型树的叶节点生成函数
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    X[:, 1: n] = dataSet[:, 0: n-1]
    y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matirx is singular, cannot do inverse')
    ws = xTx.I * (X.T * y)
    return ws, X, y

def modelLeaf(dataSet):
    ws, X, y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(yHat - y, 2))

# 如果满足停止条件，返回None和某类模型的值 如果不满足停止条件 则创建一个新的字典并将数据集分成两份，在这两份上递归调用createTree\
# 找到数据的最佳二元切分方式
def chooseBestSplit(dataSet, leafType=regLeaf, errorType=regErr, ops=(1, 4)):
    # 用户控制参数,最小误差和最少样本数
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有值相等则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # 当前误差
    S = errorType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # 寻找最佳切分点和切分值
    for featIndex in range(n - 1):
        vals = set(dataSet[:, featIndex].T.tolist()[0])
        for splitVal in vals:
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 若切分后的样本数不够 放弃
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errorType(mat0) + errorType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    # 如果误差减少不够, 退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 构建树
def createTree(dataSet, leafType=regLeaf, errorType=regErr, ops=(1, 4)):

    feat, val = chooseBestSplit(dataSet, leafType, errorType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errorType, ops)
    retTree['right'] = createTree(rSet, leafType, errorType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 树机型塌陷处理
def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


# 树的后剪枝
def prune(tree, testData):
    # 没有测试数据 对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 若有子树, 对子树剪枝
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 若没有子树 判断是否可以合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorNoMerge > errorMerge:
            print('merging!')
            return treeMean
        else:
            return tree
    else:
        return tree



# 用书回归进行预测的代码
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model ,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    # 大的去遍历左子树
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    # 否则去右子树
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat




if __name__ == '__main__':
    trainMat = mat(loadDataSet('./data/resTree/bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('./data/resTree/bikeSpeedVsIq_test.txt'))

    myTree1 = createTree(trainMat, ops=(1, 20))
    yHat1 = createForeCast(myTree1, testMat[:, 0])
    cof1 = corrcoef(yHat1, testMat[:, 1], rowvar=0)[0, 1]
    print(cof1)

    myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    cof2 = corrcoef(yHat2, testMat[:, 1], rowvar=0)[0, 1]
    print(cof2)

    yHat3 = mat(zeros((shape(testMat)[0], 1)))
    ws, X, y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    cof3 = corrcoef(yHat3, testMat[:, 1], rowvar=0)[0, 1]
    print(cof3)
    print(('done!'))