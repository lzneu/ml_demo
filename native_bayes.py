#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : native_bayes.py
# @Author: 投笔从容
# @Date  : 2018/4/20
# @Desc  : 手写朴素贝叶斯算法
from numpy import *


# 创建一个包含在所有文档中出现的不重复的词表
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


# 创建一个文档向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in vocabulary' % word)
    return returnVec


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # p(c1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 计算p(w|c)
    p0Denom = 2.0  # 拉普拉斯平滑
    p1Denom = 2.0
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1Num / p1Denom)  # log函数避免下溢出和浮点数舍入错误
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 内积求p(w|c)
    p0 = sum(p0Vec * vec2Classify) + log(1 - pClass1)
    p1 = sum(p1Vec * vec2Classify) + log(pClass1)

    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for doc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, doc))
    # 训练
    p0Vec, p1Vec, pAbusive = trainNB0(trainMat, listClasses)

    # 分类测试
    testEntry = ['stupid', 'garbage', 'dalmation']

    # 转化成向量
    vec2Classify = array(setOfWords2Vec(myVocabList, testEntry))
    preClass = classifyNB(vec2Classify, p0Vec, p1Vec, pAbusive)
    print(preClass)

# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec

# 分词去除停用词
def testParse(bigString):
    import re
    listOfToken = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 2]

# 测试垃圾邮件分类 spam 为垃圾邮件
def spamTest():
    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):
        wordList = testParse(open('./data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        wordList = testParse(open('./data/email/spam/%d.txt' % i, encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

    vocabList = createVocabList(docList)
    # 构建训练集和测试集
    trainingSet = list(range(50))
    testingSet = []
    # 随机选取测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testingSet.append(randIndex)
        del(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])

    # 训练
    p0V, p1V, pSpam = trainNB0(trainMat, trainClass)
    errorCount = 0
    # 测试
    for docIndex in testingSet:
        wordVec = setOfWords2Vec(vocabList, docList[docIndex])
        if classList[docIndex] != classifyNB(array(wordVec), p0V, p1V, pSpam):
            errorCount += 1
    print('the error rate is %f' % (float(errorCount)/len(testingSet)))


if __name__ == '__main__':
    spamTest()