package com.zhixing.nlp

import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

/**
  * Created by xiaotaop on 2018/6/12.
  */
object NlpTableStruct {

  def trainDataCosine(): StructType = {
    val fields = Array(
      StructField("q1", DataTypes.LongType, false),
      StructField("q2", DataTypes.LongType, false),
      StructField("label", DataTypes.LongType, false),
      StructField("cosine", DataTypes.DoubleType, false),
      StructField("create_time", DataTypes.StringType, false)
    )

    StructType(fields)
  }

  def trainDataDistance(): StructType = {
    val fields = Array(
      StructField("q1", DataTypes.LongType, false),
      StructField("q2", DataTypes.LongType, false),
      StructField("label", DataTypes.LongType, false),
      StructField("distance", DataTypes.DoubleType, false),
      StructField("create_time", DataTypes.StringType, false)
    )

    StructType(fields)
  }
}
