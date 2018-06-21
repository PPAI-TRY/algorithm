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
  override val OUTPUT_HOME = "output/gbt"

  override val EVALUATE_MODE = 1
  override val DEBUG = 0

  override val PCA_K = 8
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
      .setMinInstancesPerNode(5)
      .setStepSize(0.01)
      .setSubsamplingRate(1.0)
      .setMaxDepth(5)
      .setFeatureSubsetStrategy("all")
      .setImpurity("entropy")

    val pipeline = new Pipeline()
      .setStages(Array(gbt))

    val paramGrid = new ParamGridBuilder()
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(2)

    if(isEvaluate()) {
      val Array(pairTrainData, pairTestData) = pairData.randomSplit(Array(0.8, 0.2))
      evaluate(cv, pairTrainData, pairTestData)
    } else {
      cvModel = cv.fit(pairData)
    }

  }

}
