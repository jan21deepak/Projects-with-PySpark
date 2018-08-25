from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("Recommend").set("spark.ui.showConsoleProgress", "false")
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
    if sc.master[0:5] == "local" or sc.master[:5] == "spark":
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
    
def Recommend(model):
    if sys.argv[1] == "--U":
        RecommendMovies(model, moviesTitle, int(sys.argv[2]))
    elif sys.argv[1] == "--M":
        RecommendUsers(model, moviesTitle, int(sys.argv[2]))
        
def RecommendMovies(model, moviesTitle, inputUserID):
    RecommendMovie = model.recommendProducts(inputUserID, 10)
    print("针对用户id " + str(inputUserID) + " 推荐下列电影：")
    for rmd in RecommendMovie:
        print("针对用户id {0} 推荐电影 {1} 推荐评分 {2}".format(rmd[0], moviesTitle[str(rmd[1])], rmd[2]))

def RecommendUsers(model, moviesTitle, inputMovieID):
    RecommendUser = model.recommendUsers(inputMovieID, 10)
    print("针对电影id {0} 电影名 {1} 推荐下列用户 id: ".format(inputMovieID, moviesTitle[str(inputMovieID)]))
    for rmd in RecommendUser:
        print("针对用户id {0} 推荐评分 {1}".format(rmd[0], rmd[2])) 
        
def loadModel(sc):
    try:
        model = MatrixFactorizationModel.load(sc, Path + "ALSmodel")
        print("载入ALSmodel模型")
        return model
    except Exception:
        print("找不到ALSmodel模型，请先训练！")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("请输入2个参数")
        exit(-1)
    sc = CreateSparkContext()
    print("=================== 数据准备 ===================")
    moviesTitle = PrepareData(sc, "movies.csv")
    print("=================== 载入模型 ===================")
    model = loadModel(sc)
    print("=================== 进行推荐 ===================")
    Recommend(model)