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
  val MAX_ITER = 20

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

    val layers = Array[Int](5, 7, 5, 2)
    val mplc = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setLayers(layers)
      .setMaxIter(MAX_ITER)
      .setSolver("gd")
      .setStepSize(0.1)
      .setBlockSize(128)

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

    if(isEvaluate()) {
      val Array(pairTrainData, pairTestData) = pairData.randomSplit(Array(0.8, 0.2))
      evaluate(cv, pairTrainData, pairTestData)
    } else {
      cvModel = cv.fit(pairData)
    }

  }


}
