package com.zhixing.nlp

/**
  * Created by xiaotaop on 2018/6/11.
  */

case class NlpDir(home: String) {

  def trainDataDistance(): String = {
    home + "/train_data_distance"
  }

  def trainDataCosine(): String = {
    home + "/train_data_cosine"
  }

}
