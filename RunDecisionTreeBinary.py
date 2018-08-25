from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import DecisionTree
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RunDecisionTreeBinary").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf = sparkConf)
    print("master=" + sc.master)
    SetLogger(sc)
    return sc

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def extract_features(field, categoriesMap, featureEnd):
    # 提取分类特征字段 （实现OneHotEncoder的功能）
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    # 提取数值字段
    numericalFeatures = [convert_float(field) for field in field[4:featureEnd]]
    # 返回"分类特征字段" + "数值特征字段"
    return np.concatenate((categoryFeatures, numericalFeatures))

def convert_float(x):
    return (0 if x == "?" else float(x))

def extract_label(field):
    label = field[-1]
    return float(label)
        
def PrepareData(sc):
    #---------------------1. 导入并转换数据---------------------
    global Path
    if sc.master[:5] == "local" or sc.master[:5] == "spark":
        Path = "file:/Users/johnnie/pythonwork/workspace/PythonProject/data/"
    else:
        Path = "hdfs://localhost:9000/user/hduser/test/data/"
    
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path + "train.tsv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print("共计：" + str(lines.count()) + "项")     
    
    #---------------------2. 建立训练评估所需数据RDD[LabeledPoint]---------------------
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, categoriesMap, len(r) - 1)))
    
    #---------------------3. 以随机方式将数据分为3个部分并返回---------------------
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData: " + str(trainData.count()) + " validationData: " + str(validationData.count()) + " testData: " + str(testData.count()))
    
    return trainData, validationData, testData, categoriesMap

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return AUC

def trainEvaluationModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={},
                                        impurity=impurityParm, maxDepth=maxDepthParm, maxBins=maxBinsParm)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数 " + \
         " impurity = " + str(impurityParm) + \
         " maxDepth = " + str(maxDepthParm) + \
         " maxBins = " + str(maxBinsParm) + \
         " ==> 所需时间 = " + str(duration) + " 秒"\
         " 结果 AUC = " + str(AUC))
    return AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model

def showchart(df, evalparm, barData, lineData, yMin, yMax):
    ax = df[barData].plot(kind="bar", title=evalparm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalparm, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle="-", marker="o", linewidth=1.2, color="r")
    plt.show()
    
def evalParameter(trainData, validationData):
    impurityList = ["gini", "entropy"]
    maxDepthList = [3, 5, 10, 15, 20, 25]
    maxBinsList = [3, 5, 10, 50, 100, 200]
    evalparm = np.random.choice(["impurity", "maxDepth", "maxBins"], 1)[0]
    # 设置当前评估的参数
    if evalparm == 'impurity':
        # 训练评估impurity参数
        maxDepthList = np.random.choice([3, 5, 10, 15, 20, 25], 1).tolist()
        maxBinsList = np.random.choice([3, 5, 10, 50, 100, 200], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
        IndexList = impurityList[:]
    elif evalparm == 'maxDepth':
        # 训练评估maxDepth参数
        impurityList = np.random.choice(["gini", "entropy"], 1).tolist()
        maxBinsList = np.random.choice([3, 5, 10, 50, 100, 200], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
        IndexList = maxDepthList[:]
    elif evalparm == 'maxBins':
        # 训练评估maxBins参数
        impurityList = np.random.choice(["gini", "entropy"], 1).tolist()
        maxDepthList = np.random.choice([3, 5, 10, 15, 20, 25], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
        IndexList = maxBinsList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                     columns=["AUC", "Duration", "Impurity", "maxDepth", "maxBins", "Model"])
    # 显示图形
    showchart(df, evalparm, "AUC", "Duration", 0.5, 0.7)
    
def evalAllParamter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
    # 找出AUC最大的参数组合
    Smtrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smtrics[0]
    # 显示调校后最佳参数组合
    print("调校后最佳参数：" + "impurity：" + str(bestParameter[2]) +
                                        " maxDepth：" + str(bestParameter[3]) +
                                        " maxBins：" + str(bestParameter[4]) + 
                                        "\n，结果AUC = " + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]

def PredictData(sc, model, categoriesMap):
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path + "test.tsv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print("共计：" + str(lines.count()) + "项") 
    # r[0]是网址
    dataRDD = lines.map(lambda r: (r[0], extract_features(r, categoriesMap, len(r))))
    DescDict = {0: "暂时性网页(ephemeral)",
               1: "长青网页(evergreen)"}
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print("网址：" + str(data[0]) + "\n" + "==> 预测：" + str(predictResult) + " 说明：" + DescDict[predictResult] + "\n")

if __name__ == "__main__":
    print("RunDecsionTreeBinary...")
    sc = CreateSparkContext()
    print("=================== 数据准备阶段 ===================")
    trainData, validationData, testData, categoriesMap = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("=================== 训练评估阶段 ===================") 
    AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model = \
    trainEvaluationModel(trainData, validationData, "entropy", 5, 5)
    # 评估impurity、maxDepth、maxBins参数
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        evalParameter(trainData, validationData)
    # 评估所有最佳参数
    elif (len(sys.argv) == 2) and (sys.argv[1] == "-a"):
        print("-----所有参数训练评估找出最好的参数组合-----")
        model = evalAllParamter(trainData, validationData,
                           ["gini", "entropy"],
                           [3, 5, 10, 15, 20, 25],
                           [3, 5, 10, 50, 100, 200])
    print("=================== 测试阶段 ===================")
    auc = evaluateModel(model, testData)
    print(" 使用 test Data 测试最佳模型，结果AUC：" + str(auc))
    print("=================== 预测数据 ===================")
    PredictData(sc, model, categoriesMap)
    print(model.toDebugString())