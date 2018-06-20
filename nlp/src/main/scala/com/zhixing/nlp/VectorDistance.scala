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

  def editDistance(s1: Array[Long], s2: Array[Long]): Double = {
    import scala.math._

    def minimum(i1: Int, i2: Int, i3: Int) = min(min(i1, i2), i3)

    val dist = Array.tabulate(s2.length + 1, s1.length + 1) { (j, i) => if (j == 0) i else if (i == 0) j else 0 }
    for (j <- 1 to s2.length; i <- 1 to s1.length)
      dist(j)(i) = if (s2(j - 1) == s1(i - 1)) dist(j - 1)(i - 1)
      else minimum(dist(j - 1)(i) + 1, dist(j)(i - 1) + 1, dist(j - 1)(i - 1) + 1)
    dist(s2.length)(s1.length)
  }

}
