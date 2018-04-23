#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trees.py
# @Author: 投笔从容
# @Date  : 2018/4/23
# @Desc  : 手写 ID3算法 决策树分类

from numpy import *
from math import log


# 计算香农熵
def calEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算熵
    entropy = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        entropy -= p*log(p, 2)
    return entropy


# 创建训练数据集
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取出
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算信息增益 返回信息增益最大的特征索引
def chooseBestFeat(dataSet):
    numFeats = len(dataSet[0]) - 1
    # H(D)
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeat = -1

    for i in range(numFeats):
        # 构建特征可能的取值
        featList = [feat[i] for feat in dataSet]
        featList = set(featList)
        # H(D|A)
        newEntropy = 0.0
        for value in featList:
            subDataSet = splitDataSet(dataSet, i, value)
            subEntropy = calEntropy(subDataSet)
            p = len(subDataSet) / float(len(dataSet))
            newEntropy += p * subEntropy
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat


import operator
# 返回出现次数最多的类
def majorityCnt(classList):
    # 统计
    classCount = {}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序
    sorttedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorttedClassCount[0][0]


# 创建树的代码
def createTree(dataSet, labels1):
    labels = labels1[:]
    # 类别完全相同停止划分
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征返回出现次数最多的类别
    if len(classList) == 1:
        return majorityCnt(classList)
    # 迭代构建决策树
    bestFeat = chooseBestFeat(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniValues = set(featValues)
    for value in uniValues:
        # 每个特征取值构建一个树枝
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        # 复制更改引用
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
    return myTree


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    # 递归判断
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # 非叶子节点 递归
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            # 叶子节点返回类别
            else:
                classLabel = secondDict[key]
    return classLabel


# 决策树的存储
def storeTree(inputTree, filename):
    import pickle
    path = './data/'
    fw = open(path + filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    path = './data/'
    fw = open(path + filename, 'rb')
    tree = pickle.load(fw)
    fw.close()
    return tree


if __name__ == '__main__':
    # 使用决策树预测隐形眼镜类型
    fr = open('./data/lenses.txt', encoding='utf-8')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    myTree = createTree(lenses, lensesLabels)
    print(myTree)