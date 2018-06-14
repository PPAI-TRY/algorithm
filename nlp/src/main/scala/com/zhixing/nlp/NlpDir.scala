package com.zhixing.nlp

/**
  * Created by xiaotaop on 2018/6/11.
  */

case class NlpDir(home: String) {

  def trainPair(): String = {
    home + "/train_pair"
  }

  def cvTransformed(): String = {
    home + "/cv_transformed"
  }

  def prediction(): String = {
    home + "/prediction"
  }

}
