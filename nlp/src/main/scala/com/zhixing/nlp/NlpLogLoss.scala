package com.zhixing.nlp

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

/**
  * Created by xiaotaop on 2018/6/12.
  */
case class NlpLogLoss(df: DataFrame, labelCol: String, predictionCol: String, probabilityCol: String,
                      rawPredictionCol: String) {

  def logloss(): Double = {
    df.printSchema()
    df.show(10)

    val matched = df
      .filter("prediction = 1.0")
      .rdd
      .map(row => {
        val prediction = row.getAs[Double](predictionCol)
        val probability = row.getAs[DenseVector](probabilityCol).toArray(1)

        prediction == 1.0 match {
          case true => -math.log(probability)
          case false => 0.0
        }
      })

    val matchedCount = matched.map(_ => 1).reduce(_ + _)
    val sumProbability = matched.sum()

    sumProbability / matchedCount
  }

  def areaUnderRoc(): Double = {
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(rawPredictionCol)
      .setMetricName("areaUnderROC")

    evaluator.evaluate(df)
  }

  def precision(): Double = {
    val matched = df
      .rdd
      .filter(row => {
        val prediction = row.getAs[Double]("prediction")
        val label = row.getAs[Int]("label")

        prediction.toLong == label
      })

    matched.count().toDouble / df.count()
  }

}
