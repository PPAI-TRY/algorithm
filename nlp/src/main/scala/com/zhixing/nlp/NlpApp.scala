package com.zhixing.nlp

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by xiaotaop on 2018/6/10.
  */
object NlpApp {
  val CHARS_PATH = "../data/char_embed.txt"
  val WORDS_PATH = "../data/word_embed.txt"
  val QUESTIONS_PATH = "../data/question.csv"
  val TRAIN_DATA_PATH = "../data/train.csv"
  val TEST_DATA_PATH = "../data/test.csv"

  val PCA_K = 16

  val DEBUG = 1

  var AppName = getClass().getName()

  @transient lazy val logger = org.apache.log4j.LogManager.getLogger(AppName)
  @transient var conf: SparkConf = null
  @transient var sc: SparkContext = null
  @transient var sparkSession: SparkSession = null

  val OUTPUT_HOME = "output"

  var questions: RDD[Question] = null
  var wordFeatures: RDD[Word] = null
  var charFeatures: RDD[WordChar] = null
  var trainData: RDD[(Int, Long, Long)] = null  //(label, srcQuestionId, destQuestionId)
  var testData: RDD[(Long, Long)] = null  //(srcQuestionId, destQuestionId)

  var questionWordFeatures: RDD[(Long, List[Double])] = null  // (questionId, wordFeatures)
  var questionCharFeatures: RDD[(Long, List[Double])] = null  // (questionId, charFeatures)
  var trainDataDistance: RDD[(Long, Double)] = null   // (label, distance)
  var trainDataCosine: RDD[(Long, Double)] = null  // (label, cosine)
  var shrinkWordFeatures: RDD[Word] = null


  def main(args: Array[String]): Unit = {

    conf = Helper().getBaseConf(AppName)
    sc = new SparkContext(conf)
    sparkSession = SparkSession.builder().appName(AppName).config(conf).getOrCreate()

    loadData()
    initShrinkWordFeatures()
    initQuestionFeatures()
    initTrainDataDistance()
    initTrainDataCosine()

    //TODO: TRAIN
    train()

    //TODO: predict
    predict()
  }

  def loadData(): Unit = {
    charFeatures = sc.textFile(CHARS_PATH)
      .map(line => {
        val fields = line.split(" ")
        val charId = fields(0).substring(1).toLong
        var features: List[Double] = List()

        for (i <- 1 to fields.length - 1) {
          features = fields(i).toDouble +: features
        }

        assert(features.length == fields.length - 1)

        WordChar(charId, new DenseVector(features.toArray))
      })
      .cache()
    if(isDebug()) {
      logger.info(s"chars count ${charFeatures.count()}")
      charFeatures.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }

    //DONE: load words
    wordFeatures = sc.textFile(WORDS_PATH)
      .map(line => {
        val fields = line.split(" ")
        val wordId = fields(0).substring(1).toLong
        var features: List[Double] = List()

        for (i <- 1 to fields.length - 1) {
          features = fields(i).toDouble +: features
        }

        assert(features.length == fields.length - 1)

        Word(wordId, new DenseVector(features.toArray))
      })
      .cache()
    if(isDebug()) {
      logger.info(s"words count ${wordFeatures.count()}")
      wordFeatures.take(5).foreach(entry => logger.info(s" -> features length ${entry.features.toArray.length}"))
    }

    //DONE: load questions
    questions = sc.textFile(QUESTIONS_PATH)
      .filter(line => {
        val questionId = try {
          line.split(",")(0).substring(1).toLong
        } catch {
          case _ => -1L
        }

        questionId >= 0
      })
      .map(line => {
        val columns = line.split(",")
        assert(columns.length == 3)
        val questionId = columns(0).substring(1).toLong
        val wordFeatures = columns(1).split(" ").map(_.substring(1).toLong).toList
        val charFeatures = columns(2).split(" ").map(_.substring(1).toLong).toList

        Question(questionId, wordFeatures, charFeatures)
      })
      .cache()
    if(isDebug()) {
      logger.info(s"questions count ${questions.count()}")
      questions.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }

    trainData = sc.textFile(TRAIN_DATA_PATH)
      .filter(line => {
        val fields = line.split(",")
        val label = fields(0)

        label != "label"
      })
      .map(line => {
        val fields = line.split(",")
        val label = fields(0).toInt
        val srcId = fields(1).substring(1).toLong
        val destId = fields(2).substring(1).toLong

        (label, srcId, destId)
      })
      .cache()
    if(isDebug()) {
      logger.info(s"train count ${trainData.count()}")
      trainData.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }

    testData = sc.textFile(TEST_DATA_PATH)
      .filter(line => {
        val fields = line.split(",")

        fields(0) != "q1"
      })
      .map(line => {
        val fields = line.split(",")
        val srcId = fields(0).substring(1).toLong
        val destId = fields(1).substring(1).toLong

        (srcId, destId)
      })
    if(isDebug()) {
      logger.info(s"test count ${testData.count()}")
      testData.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }
  }

  def train(): Unit = {
    //TODO: select model

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setLabelCol("questionId")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(1.0, 0.5, 0.2, 0.1, 0.01))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(7)
      .setParallelism(3)

    //TODO: train model
    //val cvModel = cv.fit(wordDf)


    //TODO: evaluate model
  }

  def initQuestionFeatures(): Unit = {
    val wordIdFeaturesMap = shrinkWordFeatures
      .map(wf => {
        (wf.wid, wf.features.toArray)
      })
      .collectAsMap()

    questionWordFeatures = questions
      .map(question => {

        val sumWordFeatures = question.wordIds
          .map(wordId => {
            wordIdFeaturesMap.get(wordId) match {
              case Some(v: Array[Double]) => v
              case _ => Array(0.0)
            }
          })
          .reduce((a, b) => {
            var c: Array[Double] = Array()

            for(i <- 0 to a.length - 1) {
              val v = a(i) + b(i)
              c = v +: c
            }

            assert(c.length == a.length)

            c
          })
          .toList

        (question.qid, sumWordFeatures)
      })
    if(isDebug()) {
      questionWordFeatures.take(5).foreach(entry => logger.info(s"questionWordFeatures -> ${entry}"))
    }

  }

  def initShrinkWordFeatures(): Unit = {
    val source = wordFeatures
      .map(word => {
        (word.wid, word.features)
      })

    val sourceDf = sparkSession.createDataFrame(source).toDF("wordId", "features")

    shrinkWordFeatures = reducerDimension(sourceDf, "features", "newFeatures")
      .rdd
      .map(row => {
        val wordId = row.getAs[Long]("wordId")
        val newFeatures = row.getAs[DenseVector]("newFeatures")

        Word(wordId, newFeatures)
      })
    if(isDebug()) {
      logger.info(s"shrinkWordFeatures's features length ${shrinkWordFeatures.first().features.toArray.length}")
    }
  }

  def reducerDimension(source: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val pca = new PCA()
      .setK(PCA_K)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)

    val pcaModel = pca.fit(source)

    pcaModel.transform(source)
  }

  def initTrainDataDistance(): Unit = {
    val questionFeaturesMap = trainData
      .flatMap(item => {
        List(item._2, item._3)
      })
      .map(questionId => (questionId, 1))
      .join(questionWordFeatures)
      .map(item => {
        (item._1, item._2._2)
      })
      .collectAsMap()

    trainDataDistance = trainData
      .map(item => {
        val label = item._1
        val srcFeatures = questionFeaturesMap.get(item._2).get.toArray
        val destFeatures = questionFeaturesMap.get(item._3).get.toArray
        val srcVectors = Vectors.dense(srcFeatures)
        val destVectors = Vectors.dense(destFeatures)

        val distance = Vectors.sqdist(srcVectors, destVectors)

        (label, distance)
      })
    if(isDebug()) {
      trainDataDistance.take(5).foreach(entry => logger.info(s"trainDataDistance -> ${entry}"))
    }

    trainDataDistance.saveAsTextFile(NlpDir(OUTPUT_HOME).trainDataDistance())
  }

  def initTrainDataCosine(): Unit = {
    val questionFeaturesMap = trainData
      .flatMap(item => {
        List(item._2, item._3)
      })
      .map(questionId => (questionId, 1))
      .join(questionWordFeatures)
      .map(item => {
        (item._1, item._2._2)
      })
      .collectAsMap()

    trainDataCosine = trainData
      .map(item => {
        val label = item._1
        val srcFeatures = questionFeaturesMap.get(item._2).get.toArray
        val destFeatures = questionFeaturesMap.get(item._3).get.toArray

        val cosine = CosineSimilarity.cosineSimilarity(srcFeatures, destFeatures)

        (label, cosine)
      })
    if(isDebug()) {
      trainDataCosine.take(5).foreach(entry => logger.info(s"trainDataCosine -> ${entry}"))
    }

    trainDataCosine.saveAsTextFile(NlpDir(OUTPUT_HOME).trainDataCosine())
  }

  def predict(): Unit = {
    //TODO: use model

    //TODO: generate result

    //TODO: save to disk
  }

  def isDebug(): Boolean = {
    DEBUG == 1
  }

}
