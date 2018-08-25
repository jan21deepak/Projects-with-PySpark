from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.tree import DecisionTree
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import math

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RunDecisionTreeRegression").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf = sparkConf)
    print("master=" + sc.master)
    SetLogger(sc)
    return sc

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def extract_features(record, featureEnd):
    featureSeason = [convert_float(field) for field in record[2]]
    feature = [convert_float(field) for field in record[4:featureEnd]]
    return np.concatenate((featureSeason, feature))

def convert_float(x):
    return float(x)

def extract_label(record):
    label = record[-1]
    return float(label)
        
def PrepareData(sc):
    #---------------------1. 导入并转换数据---------------------
    global Path
    if sc.master[:5] == "local" or sc.master[:5] == "spark":
        Path = "file:/Users/johnnie/pythonwork/workspace/PythonProject/data/"
    else:
        Path = "hdfs://localhost:9000/user/hduser/test/data/"
    
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path + "hour.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(","))
    print(lines.first())
    print("共计：" + str(lines.count()) + " 项")     
    
    #---------------------2. 建立训练评估所需数据RDD[LabeledPoint]---------------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, len(r) - 1)))
    print(labelpointRDD.first())
    
    #---------------------3. 以随机方式将数据分为3个部分并返回---------------------
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    print("将数据分trainData: " + str(trainData.count()) + " validationData: " + str(validationData.count()) + " testData: " + str(testData.count()))
    
    return trainData, validationData, testData

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE = metrics.rootMeanSquaredError
    return RMSE

def trainEvaluationModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo={},
                                        impurity=impurityParm, maxDepth=maxDepthParm, maxBins=maxBinsParm)
    RMSE = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：使用参数 " + \
         " impurity = " + str(impurityParm) + \
         " maxDepth = " + str(maxDepthParm) + \
         " maxBins = " + str(maxBinsParm) + \
         " ==> 所需时间 = " + str(duration) + " 秒"\
         " 结果 RMSE = %f" %RMSE)
    return RMSE, duration, impurityParm, maxDepthParm, maxBinsParm, model

def showchart(df, evalparm, barData, lineData, yMin, yMax):
    ax = df[barData].plot(kind="bar", title=evalparm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalparm, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle="-", marker="o", linewidth=1.2, color="r")
    plt.show()
    
def evalParameter(trainData, validationData):
    impurityList = ["variance"]
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
        impurityList = np.random.choice(["variance"], 1).tolist()
        maxBinsList = np.random.choice([3, 5, 10, 50, 100, 200], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
        IndexList = maxDepthList[:]
    elif evalparm == 'maxBins':
        # 训练评估maxBins参数
        impurityList = np.random.choice(["variance"], 1).tolist()
        maxDepthList = np.random.choice([3, 5, 10, 15, 20, 25], 1).tolist()
        metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
        IndexList = maxBinsList[:]
    # 转换为Pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                     columns=["RMSE", "Duration", "Impurity", "maxDepth", "maxBins", "Model"])
    # 显示图形
    showchart(df, evalparm, "RMSE", "Duration", 0, 200)
    
def evalAllParamter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluationModel(trainData, validationData, impurity, maxDepth, maxBins)
              for impurity in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
    # 找出RMSE最大的参数组合
    Smtrics = sorted(metrics, key=lambda k: k[0])
    bestParameter = Smtrics[0]
    # 显示调校后最佳参数组合
    print("调校后最佳参数：" + "impurity：" + str(bestParameter[2]) +
                                        " maxDepth：" + str(bestParameter[3]) +
                                        " maxBins：" + str(bestParameter[4]) + 
                                        "\n，结果RMSE = " + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]

def PredictData(sc, model):
    #---------------------1. 导入并转换数据---------------------
    print("开始导入数据...")
    rawDataWithHeader = sc.textFile(Path + "hour.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(","))
    print(lines.first())
    print("共计：" + str(lines.count()) + " 项")     
    
    #---------------------2. 建立训练评估所需数据RDD[LabeledPoint]---------------------
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, len(r) - 1)))
    
    #---------------------3. 定义字典---------------------
    SeasonDict = {1:" 春 ", 2:" 夏 ", 3:" 秋 ", 4:" 冬 "}
    HildayDict = {0:" 非节假日 ", 2:" 假日 "}
    WeekDict = {0:" 一 ", 1:" 二 ", 2:" 三 ", 3:" 四 ", 4:" 五 ", 5:" 六 ", 6:" 日 "}
    WorkDayDict = {0:" 非工作日 ", 2:" 工作日 "}
    WeatherDict = {1:" 晴 ", 2:" 阴 ", 3:" 小雨 ", 4:" 大雨 "}
    
    #---------------------4. 进行预测并显示结果---------------------
    for lp in labelpointRDD.take(20):
        predict = int(model.predict(lp.features))
        label = lp.label
        features = lp.features
        result = (" 正确 " if (label == predict) else " 错误 ")
        error = math.fabs(label - predict)
        dataDesc =  " 特征：" + SeasonDict[features[0]] + "季，" + \
                    str(features[1]) + " 月，" + str(features[2]) + " 时，"  + \
                    HildayDict[features[3]] + "，" + "星期" + WeekDict[features[4]] + "," + \
                    WorkDayDict[features[5]] + "，" + WeatherDict[features[6]] + "，" + \
                    str(features[7] / 41) + " 度，" + " 体感 " + str(features[8] / 50) + " 度，" + "湿度 " + \
                    str(features[9] / 100) + "，" + "风速 " + str(features[10] / 67) + \
                    " ==> 预测结果：" + str(predict)  + "，实际：" + str(label) + result + "，误差：" + str(error)
        print(dataDesc)

if __name__ == "__main__":
    print("RunDecisionTreeRegression...")
    sc = CreateSparkContext()
    print("=================== 数据准备阶段 ===================")
    trainData, validationData, testData = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("=================== 训练评估阶段 ===================") 
    RMSE, duration, impurityParm, maxDepthParm, maxBinsParm, model = \
    trainEvaluationModel(trainData, validationData, "variance", 20, 200)
    # 评估impurity、maxDepth、maxBins参数
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        evalParameter(trainData, validationData)
    # 评估所有最佳参数
    elif (len(sys.argv) == 2) and (sys.argv[1] == "-a"):
        print("-----所有参数训练评估找出最好的参数组合-----")
        model = evalAllParamter(trainData, validationData,
                           ["variance"],
                           [3, 5, 10, 15, 20, 25],
                           [3, 5, 10, 50, 100, 200])
    print("=================== 测试阶段 ===================")
    RMSE = evaluateModel(model, testData)
    print(" 使用 test Data 测试最佳模型，结果RMSE：" + str(RMSE))
    print("=================== 预测数据 ===================")
    PredictData(sc, model)