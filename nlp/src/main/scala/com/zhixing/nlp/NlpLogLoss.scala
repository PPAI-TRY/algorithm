package com.zhixing.nlp

import org.apache.spark.sql.DataFrame

/**
  * Created by xiaotaop on 2018/6/12.
  */
case class NlpLogLoss(df: DataFrame, predictionCol: String, probCol: String) {

  def logloss(): Double = {
    df.printSchema()
    df.show(10)

    val probs = df
      .rdd
      .map(row => {
        val prediction = row.getAs[Long](predictionCol)
        val probability = row.getAs[Double](probCol)

        -math.log(probability)
      })

    val probsCount = probs.map(_ => 1).reduce(_ + _)
    val sumProbability = probs.sum()

    sumProbability / probsCount
  }

}
