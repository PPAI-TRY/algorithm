package com.zhixing.nlp

/**
  * Created by xiaotaop on 2018/6/15.
  */
object VectorDistance {

  def euclidean(v1: Array[Double], v2: Array[Double]): Double = {
    val sum = (v1 zip v2).map(item => math.pow(item._1 - item._2, 2.0)).sum
    math.pow(sum, -2.0)
  }

  def mahattan(v1: Array[Double], v2: Array[Double]): Double = {
    (v1 zip v2).map(item => math.abs(item._1 - item._2)).sum
  }

  def chebyshev(v1: Array[Double], v2: Array[Double]): Double = {
    (v1 zip v2)
      .map(item => math.abs(item._1 - item._2))
      .max
  }

}
