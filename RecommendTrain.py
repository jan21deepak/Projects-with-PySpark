from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RecommendTrain").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf = sparkConf)
    print("master=" + sc.master)
    SetLogger(sc)
    SetPath(sc)
    return sc

def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/Users/johnnie/pythonwork/workspace/PythonProject/data/"
    else:
        Path = "hdfs://localhost:9000/user/hduser/test/data/"
        
def PrepareData(sc, file):
    if file == "movies.csv":
        moviesData = sc.textFile(Path + file)
        moviesHeader = moviesData.first()
        moviesRDD = moviesData.filter(lambda x: x != moviesHeader)
        moviesTitle = moviesRDD.map(lambda line: line.split(",")[:2]).map(lambda x: (x[0], x[1])).collectAsMap()
        return moviesTitle
    elif file == "ratings.csv":
        ratingsData = sc.textFile(Path + file)
        ratings = ratingsData.map(lambda line: line.split(",")[:3])
        ratingsRDD = ratings.map(lambda x: (x[0], x[1], x[2]))
        ratingsHeader = ratingsRDD.first()
        ratingsRDD = ratingsRDD.filter(lambda x: x != ratingsHeader)
        return ratingsRDD

def SaveModel(sc):
    try:
        model.save(sc, Path + "ALSmodel")
        print("已存储Model在ALSodel")
    except Exception:
        print("Model 已存在，请先删除再存储")

if __name__ == "__main__":
    sc = CreateSparkContext()
    print("=================== 数据准备阶段 ===================")
    ratingsRDD = PrepareData(sc, "ratings.csv")
    print("=================== 训练阶段 ===================")
    print("开始 ALS 训练，参数rank=10, iterations=10, lambda=0.1")
    model = ALS.train(ratingsRDD, 10, 10, 0.1)
    print("=================== 存储 model ===================")
    SaveModel(sc)