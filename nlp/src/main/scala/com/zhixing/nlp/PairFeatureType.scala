package com.zhixing.nlp

/**
  * Created by xiaotaop on 2018/6/15.
  */
object PairFeatureType extends Enumeration {

  type PairFeatureType = Value

  val All,
  Q1Length,
  Q2Length,
  EditDistance,
  Euclidean,
  Chebyshev,
  Mahattan,
  Angel,
  Jaccard = Value


}
