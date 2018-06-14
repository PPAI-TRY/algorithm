package com.zhixing.nlp

import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

/**
  * Created by xiaotaop on 2018/6/12.
  */
object NlpTableStruct {

  def trainPair(): StructType = {
    val fields = Array(
      StructField("q1", DataTypes.LongType, false),
      StructField("q2", DataTypes.LongType, false),
      StructField("label", DataTypes.IntegerType, false),
      StructField("distance", DataTypes.DoubleType, false),
      StructField("angel", DataTypes.DoubleType, false),
      StructField("wordDiff", DataTypes.DoubleType, false),
      StructField("create_time", DataTypes.StringType, false)
    )

    StructType(fields)
  }

  def prediction(): StructType = {
    val fields = Array(
      StructField("q1", DataTypes.LongType, false),
      StructField("q2", DataTypes.LongType, false),
      StructField("prediction", DataTypes.IntegerType, false),
      StructField("p0", DataTypes.DoubleType, false),
      StructField("p1", DataTypes.DoubleType, false),
      StructField("create_time", DataTypes.StringType, false)
    )

    StructType(fields)
  }

}
