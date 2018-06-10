package com.zhixing.nlp

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
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

  val DEBUG = 1

  var AppName = getClass().getName()

  @transient lazy val logger = org.apache.log4j.LogManager.getLogger(AppName)
  @transient var conf: SparkConf = null
  @transient var sc: SparkContext = null
  @transient var sparkSession: SparkSession = null

  var questions: RDD[Question] = null
  var wordFeatures: RDD[Word] = null
  var charFeatures: RDD[WordChar] = null
  var trainData: RDD[(Int, Long, Long)] = null  //(label, srcQuestionId, destQuestionId)
  var testData: RDD[(Long, Long)] = null  //(srcQuestionId, destQuestionId)

  def main(args: Array[String]): Unit = {

    conf = Helper().getBaseConf(AppName)
    sc = new SparkContext(conf)
    sparkSession = SparkSession.builder().appName(AppName).config(conf).getOrCreate()

    loadData()

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
      wordFeatures.take(5).foreach(entry => logger.info(s" -> ${entry}"))
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

    trainData = sparkSession.read.csv(TRAIN_DATA_PATH)
      .rdd
      .map(row => {
        val label = row.getAs[Long]("label").toInt
        val srcId = row.getAs[String]("q1").substring(1).toLong
        val destId = row.getAs[String]("q2").substring(1).toLong

        (label, srcId, destId)
      })
      .cache()
    if(isDebug()) {
      logger.info(s"train count ${trainData.count()}")
      trainData.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }

    testData = sparkSession.read.csv(TEST_DATA_PATH)
      .rdd
      .map(row => {
        val srcId = row.getAs[String]("q1").substring(1).toLong
        val destId = row.getAs[String]("q2").substring(1).toLong

        (srcId, destId)
      })
    if(isDebug()) {
      logger.info(s"test count ${testData.count()}")
      testData.take(5).foreach(entry => logger.info(s" -> ${entry}"))
    }
  }

  def train(): Unit = {
    //TODO: select model
    val questionDf = initTrainDataset()

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
    val cvModel = cv.fit(wordDf)


    //TODO: evaluate model
  }

  def initTrainDataset(): DataFrame = {
    /*
    Returns:
      row -> (label, word features, char features)
     */
    val charIdFeaturesMap = charFeatures
      .map(cf => {
        (cf.charId, cf.features.toArray)
      })
      .collectAsMap()

    val wordIdFeaturesMap = wordFeatures
      .map(wf => {
        (wf.wid, wf.features.toArray)
      })
      .collectAsMap()

    val questionFeaturesMap = questions
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
              c(i) = a(i) + b(i)
            }

            c
          })

        val sumCharFeatures = question.charIds
          .map(charId => {
            charIdFeaturesMap.get(charId) match {
              case Some(v: Array[Double]) => v
              case _ => Array(0.0)
            }
          })
          .reduce((a, b) => {
            var c: Array[Double] = Array()

            for(i <- 0 to a.length - 1) {
              c(i) = a(i) + b(i)
            }

            c
          })

        (question.qid, (sumWordFeatures, sumCharFeatures))
      })
      .collectAsMap()

    val trainRdd = trainData
      .map()

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

