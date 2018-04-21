#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : native_bayes.py
# @Author: 投笔从容
# @Date  : 2018/4/20
# @Desc  : 朴素贝叶斯 算法练习

# 创建一个包含在所有文档中出现的不重复的词表
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

# 创建一个文档向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in vocabList:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in vocabulary' % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    pass

