import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RunSVMWithSGDBinary").set("spark.ui.showConsoleProgress", "false")
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
    #return float(label)
    return label
        
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
    print("标准化之前：")
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelRDD = lines.map(lambda r: extract_label(r))
    featureRDD = lines.map(lambda r: extract_features(r, categoriesMap, len(r) - 1))
    print(featureRDD.first())
    print("\n")
    print("标准化之后：")
    stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
    ScalerFeatureRDD = stdScaler.transform(featureRDD)
    print(ScalerFeatureRDD.first())
    labelpoint = labelRDD.zip(ScalerFeatureRDD)
    # r[0]是label
    # r[1]是features
    labelpointRDD = labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))
    
    #---------------------3. 以随机方式将数据分为3个部分并返回---------------------
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData: " + str(trainData.count()) + " validationData: " + str(validationData.count()) + " testData: " + str(testData.count()))
    
    return trainData, validationData, testData, categoriesMap

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    score = score.map(lambda p: float(p))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return AUC

def trainEvaluationModel(trainData, validationData, numIterations, stepSize, regParam):
    startTime = time()
    # numIterations：使用随机梯度下降法的迭代次数，默认为100
    # stepSize：随机梯度下降的步长，默认为1
    # regParam：正则化参数，数值在0~1之间
    model = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数 " + \
         " numIterations = " + str(numIterations) + \
         " stepSize = " + str(stepSize) + \
         " regParam = " + str(regParam) + \
         " ==> 所需时间 = " + str(duration) + " 秒"\
         " 结果 AUC = " + str(AUC))
    return AUC, duration, numIterations, stepSize, regParam, model

def showchart(df, evalparm, barData, lineData, yMin, yMax):
    ax = df[barData].plot(kind="bar", title=evalparm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalparm, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle="-", marker="o", linewidth=1.2, color="r")
    plt.show()
    
def evalParameter(trainData, validationData):
    numIterationsList = [1, 3, 5, 15, 25]
    stepSizeList = [10, 20, 50, 100, 200]
    regParamList = [0.01, 0.05, 0.1, 0.5, 1.0]
    evalparm = np.random.choice(["numIterations", "stepSize", "regParam"], 1)[0]
    # 设置当前评估的参数
    if evalparm == 'numIterations':
        # 训练评估numIterations参数
        stepSizeList = np.random.choice([10, 20, 50, 100, 200], 1).tolist()
        regParamList = np.random.choice([0.01, 0.05, 0.1, 0.5, 1.0], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, numIterations, stepSize, regParam)
              for numIterations in numIterationsList
              for stepSize in stepSizeList
              for regParam in regParamList]
        IndexList = numIterationsList[:]
    elif evalparm == 'stepSize':
        # 训练评估stepSize参数
        numIterationsList = np.random.choice([1, 3, 5, 15, 25], 1).tolist()
        regParamList = np.random.choice([0.01, 0.05, 0.1, 0.5, 1.0], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, numIterations, stepSize, regParam)
              for numIterations in numIterationsList
              for stepSize in stepSizeList
              for regParam in regParamList]
        IndexList = stepSizeList[:]
    elif evalparm == 'regParam':
        # 训练评估regParam参数
        numIterationsList = np.random.choice([1, 3, 5, 15, 25], 1).tolist()
        stepSizeList = np.random.choice([10, 20, 50, 100, 200], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, numIterations, stepSize, regParam)
              for numIterations in numIterationsList
              for stepSize in stepSizeList
              for regParam in regParamList]
        IndexList = regParamList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                     columns=["AUC", "Duration", "numIterations", "stepSize", "regParam", "Model"])
    # 显示图形
    showchart(df, evalparm, "AUC", "Duration", 0.5, 0.7)
    
def evalAllParamter(trainData, validationData, numIterationsList, stepSizeList, regParamList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluationModel(trainData, validationData, numIterations, stepSize, regParam)
              for numIterations in numIterationsList
              for stepSize in stepSizeList
              for regParam in regParamList]
    # 找出AUC最大的参数组合
    Smtrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smtrics[0]
    # 显示调校后最佳参数组合
    print("调校后最佳参数：" + "numIterations：" + str(bestParameter[2]) +
                                        " stepSize：" + str(bestParameter[3]) +
                                        " maxBatchFraction：" + str(bestParameter[4]) + 
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
    print("RunSVMWithSGDBinary...")
    sc = CreateSparkContext()
    print("=================== 数据准备阶段 ===================")
    trainData, validationData, testData, categoriesMap = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("=================== 训练评估阶段 ===================") 
    AUC, duration, numIterations, stepSize, regParam, model = \
    trainEvaluationModel(trainData, validationData, 10, 3, 0.2)
    # 评估numIterations, stepSize, regParam参数
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        evalParameter(trainData, validationData)
    # 评估所有最佳参数
    elif (len(sys.argv) == 2) and (sys.argv[1] == "-a"):
        print("-----所有参数训练评估找出最好的参数组合-----")
        model = evalAllParamter(trainData, validationData,
                           [1, 3, 5, 15, 25],
                           [10, 20, 50, 100, 200],
                           [0.01, 0.05, 0.1, 0.5, 1.0])
    print("=================== 测试阶段 ===================")
    auc = evaluateModel(model, testData)
    print(" 使用 test Data 测试最佳模型，结果AUC：" + str(auc))
    print("=================== 预测数据 ===================")
    PredictData(sc, model, categoriesMap)