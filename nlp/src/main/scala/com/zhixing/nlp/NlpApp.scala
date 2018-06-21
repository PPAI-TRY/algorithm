package com.zhixing.nlp

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * Created by xiaotaop on 2018/6/10.
  */
object NlpApp extends NlpBaseApp {
  override val OUTPUT_HOME = "output/lr"

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

    val lr = new LogisticRegression()
      .setMaxIter(LR_MAX_ITER)
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setStandardization(true)
      .setElasticNetParam(0.95)
      .setRegParam(0.5)
      .setFitIntercept(true)

    val pipeline = new Pipeline()
      .setStages(Array(lr))

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
