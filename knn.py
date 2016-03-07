#-*-encoding: utf-8-*-
#author:	xiaohu
#date:	2016/3/5
#desc:	moudule of knn
import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def creatDataSet():
    group = array([[1.0,1.1], [1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#Inx:	input vector need to be classfy
#SampleMat:	Sample data mat(m x n)
#SampleLables:	sample data lables(m x 1)
#k:	k-nn paramter k
def classfy_knn(Inx, SampleMat, SampleLables, k):
    """classfy knn"""
    m = SampleMat.shape[0]
    diffMat = SampleMat - tile(Inx, (m,1))
    diffMat = diffMat**2
    diffDistances = diffMat.sum(1)**0.5
    distanceArgs = diffDistances.argsort()
    returnLable = {}
    for i in range(k):
        topilable = SampleLables[distanceArgs[i]]
#        print topilable
        returnLable[topilable] = returnLable.get(topilable,0) + 1
    returnLable = sorted(returnLable.iteritems(), key=operator.itemgetter(1), reverse = True)
    return returnLable[0][0]

def file2matrix(filename):
    """transform file to matrix"""
    fp = open(filename)
    file_lines = fp.readlines()
    m = len(file_lines)
    lable_trans = {'didntLike':1,'smallDoses':2,'largeDoses':3}
    group = zeros((m,3))
    lables = []
    index = 0
    for line in file_lines:
        line = line.strip()
        line_list = line.split('\t')
        group[index,:] = line_list[0:3]
        lables.append(int(lable_trans[line_list[-1]]))
        index += 1
    return group, lables

def autonorm(dataMat):
    m = dataMat.shape[0]
    mat_range = dataMat.max(0) - dataMat.min(0)
    mat_min = dataMat.min(0)
    newdataMat = dataMat - tile(mat_min,(m,1))
    newdataMat = newdataMat / mat_range
    return newdataMat, mat_range, mat_min
    
def datingClassTest():
    hotRatio = 0.10
    group, lables = file2matrix('datingTestSet.txt')
    group_norm, mat_range, mat_min = autonorm(group)
    m = group_norm.shape[0]
    test_num = int(m*hotRatio)
    error_count = 0.0
    for i in range(test_num):
        pre_lable = classfy_knn(group_norm[i,:], group_norm[test_num:m,:], lables[test_num:m],3)
        print "the pre_lable is %d , the real lable is %d" %(pre_lable, lables[i])
        if pre_lable != lables[i]:
            error_count += 1.0
    print "the total error rate is %f" %(error_count/float(test_num))

def classfyPerson():
    result_list = ['didntLike','smallDoses','largeDoses']
    percentTats = float (raw_input("percentage of time playing vediogames?"))
    ffMiles = float (raw_input("freguent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    group, lables = file2matrix('datingTestSet.txt')
    group_norm, mat_range, mat_min = autonorm(group)
    inArry = array( [ffMiles, percentTats, iceCream] )
    pre_lable = classfy_knn((inArry-mat_min)/mat_range, group_norm, lables, 3)
    print "you may %s" %result_list[pre_lable-1]
    return group_norm, mat_range, mat_min 

def image2vector(filename):
    fp = open(filename)
    file_lines = fp.readlines()
    return_vec = zeros((1,1024))
    for i in range(32):
        for j in range(32):
            return_vec[0,32*i + j] = int(file_lines[i][j])
    return return_vec

def handingClassTest():
    hwLables = []
    trainFilelist = listdir('./digits/trainingDigits')
    mTrain = len(trainFilelist)
    train_mat = zeros((mTrain,32*32))
    for i in range(mTrain):
        filestr = trainFilelist[i].split('.')[0]
        realLable = int(filestr.split('_')[0])
#        print 'realLable: %d' %realLable
        hwLables.append(realLable)
        train_mat[i,:] = image2vector('./digits/trainingDigits/%s'%trainFilelist[i])
#        print 'digits/trainingDigits/%s' %trainFilelist[i]
    testFilelist = listdir('./digits/testDigits')
    mTest = len(testFilelist)
    error_count = 0.0
    for i in range(mTest):
        filestr = testFilelist[i].split('.')[0]
        realLable = int(filestr.split('_')[0])
        testimg_vector = image2vector('./digits/testDigits/%s'%testFilelist[i])
        pre_Lable = classfy_knn(testimg_vector, train_mat, hwLables, 5)
        print "the pre_lable is:%d, the real_lable is:%d" %(pre_Lable,realLable)
        if pre_Lable != realLable:
            error_count += 1.0
    print "the total number of errors is :%d" %error_count
    print "the total error rate is: %f" % (error_count/float(mTest))
#    testimg_vector = image2vector('digits/trainingDigits/5_0.txt') 
#    print testimg_vector  
#    pre_Lable = classfy_knn(testimg_vector, train_mat, hwLables, 10)
#    print "the pre_lable is:%d, the real_lable is:%d" %(pre_Lable,5)
