package com.zhixing.nlp

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

/**
  * Created by xiaotaop on 2018/6/15.
  */
object NlpGbtApp extends NlpBaseApp {

  override val EVALUATE_MODE = 1
  override val PCA_K = 16
  override val LR_MAX_ITER = 10


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

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(LR_MAX_ITER)
      .setFeatureSubsetStrategy("sqrt")
      .setImpurity("entropy")

    val pipeline = new Pipeline()
      .setStages(Array(gbt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.minInstancesPerNode, Array(1, 3, 5))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)
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

      print(conclusion)

      if(isDebug()) {
        logger.info(conclusion)
      }
    }

    //DONE: train model
    cvModel = cv.fit(pairData)

  }

}
