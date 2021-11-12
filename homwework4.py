# Grupo 117 Aprendizagem HomeWork 4
# Bernardo Castico ist196845
# Hugo Rita ist196870

import pandas as pd
import sklearn
from scipy.io import arff
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

def getDataToMatrix(lines):
    realLines = []
    data = []
    toDelete = []
    for i in range(len(lines)):
        if i > 11:
            realLines += [lines[i]]
    for i in range(len(realLines)):
        for j in range(len(realLines[i])):
            if realLines[i][j] == "benign\n":
                realLines[i][j] = "benign"
            elif realLines[i][j] == "malignant\n":
                realLines[i][j] = "malignant"
            elif realLines[i][j] == '?':
                toDelete += [i]
            else:
                realLines[i][j] = int(realLines[i][j])
    for i in range(len(realLines)):
        if i not in toDelete:
            data += [realLines[i]]
    return data

def splitData(list):
    a = []
    b = []
    for i in list:
        a.append(i[:-1])
        b.append(i[-1])
    return [a,b]

def main():
    data, res2 = [],[]
    cluster02, cluster12, cluster03, cluster13, cluster23, cluster05, cluster15, cluster25 = 0,0,0,0,0,0,0,0
    Benign02, malignant02, Benign12, malignant12, Benign03, malignant03, Benign13, malignant13, Benign23, malignant23 = 0,0,0,0,0,0,0,0,0,0
    Benign05, malignant05, Benign15, malignant15, Benign25, malignant25 = 0,0,0,0,0,0
    xCluster0, yCluster0, xCluster1, yCluster1, xCluster2, yCluster2 = [],[],[],[],[],[]
    with open("HW3-breast.txt") as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split(',')
        res2.append(tmp)
    data = getDataToMatrix(res2)

    trainDataSplit = splitData(data)

    kMeans2 = KMeans(n_clusters=2, random_state=0).fit(trainDataSplit[0])
    kMeans3 = KMeans(n_clusters=3, random_state=0).fit(trainDataSplit[0])

    kLabels2 = kMeans2.labels_
    kLabels3 = kMeans3.labels_

    for i in range(len(kLabels2)):
        if kLabels2[i] == 0:
            cluster02 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant02 += 1
            else:
                Benign02 += 1
        elif kLabels2[i] == 1:
            cluster12 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant12 += 1
            else:
                Benign12 += 1
        if kLabels3[i] == 0:
            cluster03 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant03 += 1
            else:
                Benign03 += 1
        elif kLabels3[i] == 1:
            cluster13 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant13 += 1
            else:
                Benign13 += 1
        elif kLabels3[i] == 2:
            cluster23 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant23 += 1
            else:
                Benign23 += 1

    ECR2 = 0.5*((cluster02-max(Benign02,malignant02)) + (cluster12-max(Benign12, malignant12)))
    ECR3 = (1/3)*((cluster03-max(Benign03,malignant03)) + (cluster13-max(Benign13, malignant13))+ (cluster23-max(Benign23, malignant23)))

    print("ECR K = 2")
    print(ECR2)
    print("ECR k = 3")
    print(ECR3)
    print("Silhouette K = 2")
    print(silhouette_score(trainDataSplit[0], kLabels2))
    print("Silhouette K = 3")
    print(silhouette_score(trainDataSplit[0], kLabels3))

    #EX5

    decision = SelectKBest(mutual_info_classif, k=2).fit(trainDataSplit[0], trainDataSplit[1])
    decisionTrainData = decision.transform(trainDataSplit[0])

    kMeans3Ex5 = KMeans(n_clusters=3, random_state=0).fit(decisionTrainData)
    kLabelsEx5 = kMeans3Ex5.labels_

    for i in range(len(kLabelsEx5)):
        if kLabelsEx5[i] == 0:
            cluster05 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant05 += 1
            else:
                Benign05 += 1
        elif kLabelsEx5[i] == 1:
            cluster15 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant15 += 1
            else:
                Benign15 += 1
        elif kLabelsEx5[i] == 2:
            cluster25 += 1
            if trainDataSplit[1][i] == 'malignant':
                malignant25 += 1
            else:
                Benign25 += 1

    ECR5 = (1/3)*((cluster05-max(Benign05,malignant05)) + (cluster15-max(Benign15, malignant15))+ (cluster25-max(Benign25, malignant25)))
    print("ECR Ex5")
    print(ECR5)
    print("Silhouette Ex5")
    print(silhouette_score(decisionTrainData, kLabelsEx5))

    for i in range(len(kLabelsEx5)):
        if kLabelsEx5[i] == 0:
            xCluster0 += [decisionTrainData[i][0]]
            yCluster0 += [decisionTrainData[i][1]]
        elif kLabelsEx5[i] == 1:
            xCluster1 += [decisionTrainData[i][0]]
            yCluster1 += [decisionTrainData[i][1]]
        else:
            xCluster2 += [decisionTrainData[i][0]]
            yCluster2 += [decisionTrainData[i][1]]

    plt.scatter(xCluster0, yCluster0, label="Cluster 0")
    plt.scatter(xCluster1, yCluster1, label="Cluster 1")
    plt.scatter(xCluster2, yCluster2, label="Cluster 2")

    plt.xlabel('x - BestFeature1')
    plt.ylabel('y - BestFeature2')
    plt.title('Cluster solution with k=3 and 2 K best features')

    # show a legend on the plot
    plt.legend()
    plt.show()

main()