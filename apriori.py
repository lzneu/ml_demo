#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : apriori.py
# @Author: 投笔从容
# @Date  : 2018/5/8
# @Desc  : 关联规则


from numpy import *


def loadData():
    return [
        [1,3,4],
        [2,3,5],
        [1,2,3,5],
        [2,5]
    ]


#  创建单个元素的候选项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append(item)
    C1.sort()
    return map(frozenset, C1)   # 冰冻集合


def scanD(D, Ck, minSupport):
    '''
    :param D: 数据集
    :param Ck: 第K次的 候选频繁项集
    :param minSupport: 最小支持度阈值
    :return: 判断后的候选集 及其 支持度
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] == 1
    numIterms = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numIterms
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    '''
    构建k+1项 的候选项集Ck+1
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return:
    '''
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            # 前k-2个项相同时 将两个集合合并
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList


# apriori算法
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supportK = scanD(D, Ck, minSupport)
        supportData.update(supportK)
        L.append(Lk)
        k += 1
    return L, supportData


