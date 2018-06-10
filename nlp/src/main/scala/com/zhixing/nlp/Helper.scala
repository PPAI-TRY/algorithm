package com.zhixing.nlp

import org.apache.spark.SparkConf

/**
  * Created by xiaotaop on 2018/6/10.
  */
case class Helper() {

  def getBaseConf(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.rdd.compress", "true")

    conf
  }

}
