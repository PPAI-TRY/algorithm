#!/bin/sh

. ./commons.sh


SPARK_HOME="/Users/xiaotaop/Documents/software/spark-2.3.0-bin-hadoop2.7"
SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"

DJ_CMD="/Users/xiaotaop/Documents/pyenvs/dj18/bin/python /Users/xiaotaop/Documents/gitroom/recsys/similar_album/similar/manage.py"

LIB="/Users/xiaotaop/Documents/software/spark-2.0.1-local/jars/mysql-connector-java-5.1.40-bin.jar"
LOG4J_CONF="${SPARK_HOME}/conf/log4j.properties"

CLASS="com.zhixing.nlp.NlpApp"
JAR="/Users/xiaotaop/Documents/gitroom/ppai-algorithm/nlp/target/scala-2.11/nlp_2.11-1.0.jar"

EXECUTOR_MEMORY="6G"
EXECUTOR_CORES=3


function build() {
    sbt package
}


function run() {
    nohup ${SPARK_SUBMIT} \
        --master local[1] \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${EXECUTOR_MEMORY} \
        --driver-cores ${EXECUTOR_CORES} \
        --class ${CLASS} \
        --conf spark.memory.fraction="0.8" \
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
