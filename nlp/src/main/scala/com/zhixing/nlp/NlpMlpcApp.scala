package com.zhixing.nlp

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiaotaop on 2018/6/20.
  */
object NlpMlpcApp extends NlpBaseApp {
  override val OUTPUT_HOME = "output/lmpc"

  override val EVALUATE_MODE = 1
  override val DEBUG = 0

  override val PCA_K = 8
  val MAX_ITER = 10

  def main(args: Array[String]): Unit = {

    conf = Helper().getBaseConf(AppName)
    sc = new SparkContext(conf)
    sparkSession = SparkSession.builder().appName(AppName).config(conf).getOrCreate()

    loadData()
    initShrinkWordFeatures()
    initQuestionFeatures()

    initTrainQuestionPairFeatures()
    train()

    if (isEvaluate() == false) {
      predict()
    }
  }

  def train(): Unit = {
    val pairData = sparkSession.createDataFrame(trainQuestionPairFeatures).toDF("label", "features")

    val layers = Array[Int](5, 2)
    val mplc = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setLayers(layers)
      .setMaxIter(MAX_ITER)

    val pipeline = new Pipeline()
      .setStages(Array(mplc))

    val paramGrid = new ParamGridBuilder()
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(2)

    //evaluate model
    if(isEvaluate()) {
      val Array(pairTrainData, pairTestData) = pairData.randomSplit(Array(0.8, 0.2))

      val evaluateModel = cv.fit(pairTrainData)
      val evaluateResult = evaluateModel.transform(pairTestData)
      evaluateResult.write.mode("overwrite").json(NlpDir(OUTPUT_HOME).cvTransformed())

      val evaluator = NlpLogLoss(evaluateResult, "label", "prediction", "probability", "rawPrediction")
      val logLoss = evaluator.logloss()
      val areaUnderRoc = evaluator.areaUnderRoc()
      val precision = evaluator.precision()
      val conclusion = s"logloss, areaUnderRoc, precision, | ${logLoss} | ${areaUnderRoc} | ${precision} | | |"

      println(conclusion)
      logger.info(conclusion)
    }

    //DONE: train model
    cvModel = cv.fit(pairData)

  }


}
