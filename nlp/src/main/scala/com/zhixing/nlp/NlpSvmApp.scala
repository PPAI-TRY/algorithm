package com.zhixing.nlp

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiaotaop on 2018/6/20.
  */
object NlpSvmApp extends NlpBaseApp {
  override val OUTPUT_HOME = "output/svm"

  override val PCA_K = 8
  override val LR_MAX_ITER = 10

  override val EVALUATE_MODE = 1
  override val DEBUG = 0


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
    val Array(pairTrainData, pairTestData) = pairData.randomSplit(Array(0.8, 0.2))

    val svm = new LinearSVC()
      .setMaxIter(LR_MAX_ITER)
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setStandardization(true)
      .setFitIntercept(true)

    val pipeline = new Pipeline()
      .setStages(Array(svm))

    val paramGrid = new ParamGridBuilder()
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(8)
      .setParallelism(4)

    //evaluate model
    if(isEvaluate()) {
      val evaluateModel = cv.fit(pairTrainData)
      val evaluateResult = evaluateModel.transform(pairTestData)
      evaluateResult.write.mode("overwrite").json(NlpDir(OUTPUT_HOME).cvTransformed())

      val evaluator = NlpLogLoss(evaluateResult, "label", "prediction", "probability", "rawPrediction")
      val logLoss = evaluator.logloss()
      val areaUnderRoc = evaluator.areaUnderRoc()
      val precision = evaluator.precision()
      val conclusion = s"logloss, areaUnderRoc, precision, | ${logLoss} | ${areaUnderRoc} | ${precision} | | |"

      print(conclusion)
      logger.info(conclusion)
    }

    //DONE: train model
    cvModel = cv.fit(pairData)

  }

}
