import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RunNaiveBayesBinary").set("spark.ui.showConsoleProgress", "false")
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
    # NaiveBayes数值特征字段一定是大于0，所以负数转换为0
    ret = (0 if x == "?" else float(x))
    return (0 if ret < 0 else ret)

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
    stdScaler = StandardScaler(withMean=False, withStd=True).fit(featureRDD)
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

def trainEvaluationModel(trainData, validationData, lambdaParam):
    startTime = time()
    # lambda 设置lambda参数，默认值为1.0
    model = NaiveBayes.train(trainData, lambdaParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数 " + \
         " lambda = " + str(lambdaParam) + \
         " ==> 所需时间 = " + str(duration) + " 秒"\
         " 结果 AUC = " + str(AUC))
    return AUC, duration, lambdaParam, model

def showchart(df, barData, lineData, yMin, yMax):
    ax = df[barData].plot(kind="bar", title="lambda", figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel("lambda", fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle="-", marker="o", linewidth=1.2, color="r")
    plt.show()
    
def evalParameter(trainData, validationData):
    lambdaParamList = [1.0, 3.0, 5.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 100.0]
    # 训练评估lambda参数
    metrics = [trainEvaluationModel(trainData, validationData, lambdaParam)
              for lambdaParam in lambdaParamList]
    IndexList = lambdaParamList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                     columns=["AUC", "Duration", "lambdaParam", "Model"])
    # 显示图形
    showchart(df, "AUC", "Duration", 0.5, 0.7)
    
def evalAllParamter(trainData, validationData, lambdaParamList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluationModel(trainData, validationData, lambdaParam)
              for lambdaParam in lambdaParamList]
    # 找出AUC最大的参数组合
    Smtrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smtrics[0]
    # 显示调校后最佳参数组合
    print("调校后最佳参数：" + "lambdaParam：" + str(bestParameter[2]) + 
                                        "\n，结果AUC = " + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[3]

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
    print("RunNaiveBayesBinary...")
    sc = CreateSparkContext()
    print("=================== 数据准备阶段 ===================")
    trainData, validationData, testData, categoriesMap = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("=================== 训练评估阶段 ===================") 
    AUC, duration, lambdaParam, model = \
    trainEvaluationModel(trainData, validationData, 25.0)
    # 评估lambdaParam参数
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        evalParameter(trainData, validationData)
    # 评估最佳参数
    elif (len(sys.argv) == 2) and (sys.argv[1] == "-a"):
        print("-----所有参数训练评估找出最好的参数组合-----")
        model = evalAllParamter(trainData, validationData,
                                [1.0, 3.0, 5.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 100.0])
    print("=================== 测试阶段 ===================")
    auc = evaluateModel(model, testData)
    print(" 使用 test Data 测试最佳模型，结果AUC：" + str(auc))
    print("=================== 预测数据 ===================")
    PredictData(sc, model, categoriesMap)