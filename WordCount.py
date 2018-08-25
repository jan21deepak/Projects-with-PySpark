#_*_ coding: utf-8 _*_

from pyspark import SparkContext, SparkConf

def CreateSparkContext():
    # 创建SparkConf配置设置对象
    # sparkConf = SparkConf() \
            # 设置App名称，此App名称会显示在Spark或Hadoop YARN UI界面
    #        .setAppName("WordCounts") \
            # 设置不要显示Spark执行速度，以免屏幕显示界面太乱
    #       .set("spark.ui.showConsoleProgress", "false")
    sparkConf = SparkConf().setAppName("WordCounts").set("spark.ui.showConsoleProgress", "false")
    # 创建SparkContext传入参数：SparkConf配置设置对象
    sc = SparkContext(conf = sparkConf)
    # 显示当前运行模式：local、YARN client或Spark Stand alone
    print("master=" + sc.master)
    # 设置不要显示太多信息
    SetLogger(sc)
    # 配置文件读取路径
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
        
if __name__ == "__main__":
    print("开始执行RunWordCount...")
    sc = CreateSparkContext()
    print("开始读取文本文件....")
    textFile = sc.textFile(Path + "test.txt")
    print("文本文件共 " + str(textFile.count()) + " 行")
    countsRDD = textFile.flatMap(lambda line: line.split(' ')).map(lambda x:(x, 1)).reduceByKey(lambda x, y: x + y)
    print("文字统计共 " + str(countsRDD.count()) + " 项数据")
    print("开始保存至文本文件...")
    try:
        countsRDD.saveAsTextFile(Path + "output/")
    except Exception as e:
        print("输出目录已经存在，请先删除原有目录")
    sc.stop()