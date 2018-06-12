package com.zhixing.nlp

import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, TimeZone}

import org.apache.spark.SparkConf

/**
  * Created by xiaotaop on 2018/6/10.
  */
case class Helper() {
  val defaultTimezone = TimeZone.getTimeZone("GMT+8")

  def getBaseConf(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.rdd.compress", "true")

    conf
  }

  def getNow(): Date = {
    val calendar = Calendar.getInstance(defaultTimezone)

    calendar.getTime()
  }

  def string2Date(src: String, format: String = "yyyy-MM-dd HH:mm:ss"): Option[Date] = {
    src.isEmpty match {
      case true => null
      case false =>
        val formatter = new SimpleDateFormat(format)
        Some(formatter.parse(src))
    }
  }

  def date2String(dt: Option[Date], format: String = "yyyy-MM-dd HH:mm:ss"): String = {
    dt match {
      case Some(d: Date) =>
        val formatter = new SimpleDateFormat(format)
        formatter.format(d)
      case _ => ""
    }
  }

  def timestamp2Date(ts: java.sql.Timestamp): Date = {
    val cal = Calendar.getInstance(defaultTimezone)

    cal.setTimeInMillis(ts.getTime)
    cal.getTime
  }

  def date2timestamp(dt: Option[Date]): Option[Timestamp] = {
    dt match {
      case Some(d: Date) => Some(new Timestamp(d.getTime))
      case _ => Some(null)
    }
  }


}
