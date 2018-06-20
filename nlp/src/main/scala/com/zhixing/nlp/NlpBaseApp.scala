package com.zhixing.nlp

import com.zhixing.nlp.PairFeatureType.PairFeatureType
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by xiaotaop on 2018/6/15.
  */
class NlpBaseApp extends Serializable {

  val CHARS_PATH = "../data/char_embed.txt"
  val WORDS_PATH = "../data/word_embed.txt"
  val QUESTIONS_PATH = "../data/question.csv"
  val TRAIN_DATA_PATH = "../data/train.csv"
  val TEST_DATA_PATH = "../data/test.csv"

  val PCA_K = 64
  val LR_MAX_ITER = 200

  val EVALUATE_MODE = 0

  val DEBUG = 1

  var AppName = getClass().getName()

  @transient lazy val logger = org.apache.log4j.LogManager.getLogger(AppName)
  @transient var conf: SparkConf = null
  @transient var sc: SparkContext = null
  @transient var sparkSession: SparkSession = null

  val OUTPUT_HOME = "output"

  val DEFAULT_PAIR_FEATURE_TYPE = PairFeatureType.All

  var questions: RDD[Question] = null
  var wordFeatures: RDD[Word] = null
  var charFeatures: RDD[WordChar] = null
  var trainData: RDD[(Int, Long, Long)] = null  //(label, srcQuestionId, destQuestionId)
  var testData: RDD[(Long, Long)] = null  //(srcQuestionId, destQuestionId)

  var questionWordFeatures: RDD[(Long, List[Double])] = null  // (questionId, wordFeatures)
  var questionCharFeatures: RDD[(Long, List[Double])] = null  // (questionId, charFeatures)
  var shrinkWordFeatures: RDD[Word] = null

  var trainQuestionPairFeatures: RDD[(Int, DenseVector)] = null

  var cvModel: CrossValidatorModel = null


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
      .cache()
    if(isDebug()) {
      logger.info(s"shrinkWordFeatures's features length ${shrinkWordFeatures.first().features.toArray.length}")
    }

    wordFeatures.unpersist()
  }

  def reducerDimension(source: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val pca = new PCA()
      .setK(PCA_K)
      .setInputCol(inputCol)
      .setOutputCol(outputCol)

    val pcaModel = pca.fit(source)

    pcaModel.transform(source)
  }

  def extractQuestionPairFeatures(sourcePairs: RDD[(Int, Long, Long)],
                                  featureType: PairFeatureType = DEFAULT_PAIR_FEATURE_TYPE):
    RDD[(Long, Long, Int, Array[Double])] = {
    /*
    very importance

    Params:
      sourcePairs: (label, q1, q2)
    Returns:
      row is (q1, q2, label, features)
     */

    val questionFeaturesMap = questionWordFeatures
      .collectAsMap()

    val questionIdMap = questions
      .map(q => (q.qid, q))
      .collectAsMap()

    val angels = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val srcFeatures = questionFeaturesMap.get(q1).get.toArray
        val destFeatures = questionFeaturesMap.get(q2).get.toArray
        val cosine = CosineSimilarity.cosineSimilarity(srcFeatures, destFeatures)

        ((q1, q2), Array((label, cosine, PairFeatureType.Angel)))
      })

    val jaccards = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val q1Words = questionIdMap.get(q1).get.wordIds
        val q2Words = questionIdMap.get(q2).get.wordIds
        val unionCount = (q1Words ++ q2Words).toSet.size
        val intersectionCount = q1Words.intersect(q2Words).length
        val jaccardIndex = intersectionCount.toDouble / unionCount

        ((q1, q2), Array((label, jaccardIndex, PairFeatureType.Jaccard)))
      })

    val chebyshevs = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val srcFeatures = questionFeaturesMap.get(q1).get.toArray
        val destFeatures = questionFeaturesMap.get(q2).get.toArray
        val distance = VectorDistance.chebyshev(srcFeatures, destFeatures)

        ((q1, q2), Array((label, distance, PairFeatureType.Chebyshev)))
      })

    val euclideans = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val srcFeatures = questionFeaturesMap.get(q1).get.toArray
        val destFeatures = questionFeaturesMap.get(q2).get.toArray
        val distance = VectorDistance.euclidean(srcFeatures, destFeatures)

        ((q1, q2), Array((label, distance, PairFeatureType.Euclidean)))
      })

    val mahattans = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val srcFeatures = questionFeaturesMap.get(q1).get.toArray
        val destFeatures = questionFeaturesMap.get(q2).get.toArray
        val distance = VectorDistance.mahattan(srcFeatures, destFeatures)

        ((q1, q2), Array((label, distance, PairFeatureType.Mahattan)))
      })

    val editDistances = sourcePairs
      .map(item => {
        val label = item._1
        val q1 = item._2
        val q2 = item._3
        val srcFeatures = questionIdMap.get(q1).get.wordIds.toArray
        val destFeatures = questionIdMap.get(q2).get.wordIds.toArray
        val distance = VectorDistance.editDistance(srcFeatures, destFeatures)

        ((q1, q2), Array((label, distance, PairFeatureType.EditDistance)))
      })

    val dest = angels
      .union(jaccards)
      .union(chebyshevs)
      .union(euclideans)
      .union(mahattans)
      .union(editDistances)
      .reduceByKey(_ ++ _)
      .map(item => {
        val q1 = item._1._1
        val q2 = item._1._2
        val q1WordLength = questionIdMap.get(q1).get.wordIds.length
        val q2WordLength = questionIdMap.get(q2).get.wordIds.length
        val label = item._2(0)._1
        val angel = item._2.find(_._3 == PairFeatureType.Angel).get._2
        val jaccardIndex = item._2.find(_._3 == PairFeatureType.Jaccard).get._2
        val euclidean = item._2.find(_._3 == PairFeatureType.Euclidean).get._2
        val mahattan = item._2.find(_._3 == PairFeatureType.Mahattan).get._2
        val chebyshev = item._2.find(_._3 == PairFeatureType.Chebyshev).get._2
        val editDistance = item._2.find(_._3 == PairFeatureType.EditDistance).get._2

        val features = featureType match {
          case PairFeatureType.All =>
            Array(q1WordLength, q2WordLength, angel, chebyshev, jaccardIndex, mahattan, euclidean,
              editDistance)
          case PairFeatureType.Angel => Array(angel)
          case PairFeatureType.Chebyshev => Array(chebyshev)
          case PairFeatureType.Jaccard => Array(jaccardIndex)
          case PairFeatureType.Mahattan => Array(mahattan)
          case PairFeatureType.Euclidean => Array(euclidean)
        }

        (q1, q2, label, features)
      })
      .cache()
    if(isDebug()) {
      logger.info(s"sourcePair count ${sourcePairs.count()}, dest count ${dest.count()}")
      dest.take(5).foreach(e => logger.info(s" --> ${e}"))
    }

    dest
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
            (a zip b).map(e => e._1 + e._2)
          })
          .toList

        (question.qid, sumWordFeatures)
      })
    if(isDebug()) {
      questionWordFeatures.take(5).foreach(entry => logger.info(s"questionWordFeatures -> ${entry}"))
    }
  }


  def initTrainQuestionPairFeatures(featureType: PairFeatureType = DEFAULT_PAIR_FEATURE_TYPE): Unit = {
    val featuers = extractQuestionPairFeatures(trainData, featureType)
    trainQuestionPairFeatures = featuers
      .map(item => {
        (item._3, new DenseVector(item._4))
      })
    if(isDebug()) {
      logger.info(s"trainQuestionPairFeatures count ${trainQuestionPairFeatures.count()}")
      trainQuestionPairFeatures.take(5).foreach(entry => logger.info(s" --> ${entry}"))
    }

  }

  def isDebug(): Boolean = {
    DEBUG == 1
  }

  def isEvaluate(): Boolean = {
    EVALUATE_MODE == 1
  }

  def predict(): Unit = {
    val testPairData = testData
      .map(item => (-1, item._1, item._2))

    val testPairFeatures = extractQuestionPairFeatures(testPairData)
      .map(item => {
        // (q1, q2, label, features)
        (item._1, item._2, item._3, Vectors.dense(item._4))
      })

    val testPairDf = sparkSession.createDataFrame(testPairFeatures)
      .toDF("q1", "q2", "label", "features")

    val testPredictions = cvModel.transform(testPairDf)
    testPredictions.show(10)

    //TODO: save to disk
    val toSaveRdd = testPredictions
      .rdd
      .map(row => {
        val q1 = row.getAs[Long]("q1")
        val q2 = row.getAs[Long]("q2")
        val prediction = row.getAs[Double]("prediction").toInt
        val probability = row.getAs[DenseVector]("probability").toArray
        val p0 = probability(0)
        val p1 = probability(1)
        val createAt = Helper().date2String(Some(Helper().getNow()))

        Row(q1, q2, prediction, p0, p1, createAt)
      })
    sparkSession.createDataFrame(toSaveRdd, NlpTableStruct.prediction())
      .write
      .mode("overwrite")
      .json(NlpDir(OUTPUT_HOME).prediction())

  }
}
