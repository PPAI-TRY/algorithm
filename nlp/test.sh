#!/bin/sh

. ./commons.sh


SPARK_HOME="/Users/xiaotaop/Documents/software/spark-2.3.0-bin-hadoop2.7"
SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"

DJ_CMD="/Users/xiaotaop/Documents/pyenvs/dj18/bin/python /Users/xiaotaop/Documents/gitroom/recsys/similar_album/similar/manage.py"

LIB="/Users/xiaotaop/Documents/software/spark-2.0.1-local/jars/mysql-connector-java-5.1.40-bin.jar"
LOG4J_CONF="${SPARK_HOME}/conf/log4j.properties"

CLASS="com.zhixing.nlp.NlpApp"
JAR="/Users/xiaotaop/Documents/gitroom/ppai-algorithm/nlp/target/scala-2.11/nlp_2.11-1.0.jar"

EXECUTOR_MEMORY="1G"


function build() {
    sbt compile package
}


function run() {
    nohup ${SPARK_SUBMIT} \
        --master local[2] \
        --executor-memory ${EXECUTOR_MEMORY} \
        --class ${CLASS} \
        ${JAR} &
    spin $!
}


case $1 in
build)
    build
;;
run)
    build
    run
;;
*)
    echo "Usage: ./test.sh [build]"
;;
esac
