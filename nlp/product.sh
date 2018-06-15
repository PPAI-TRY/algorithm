#!/bin/sh

. ./commons.sh

SBT_BIN="/usr/local/sbt/bin/sbt"
SPARK_SUBMIT="/usr/local/spark-2.3.0-bin-hadoop2.7/bin/spark-submit"

CLASS="com.zhixing.nlp.NlpApp"
GBT_CLASS="com.zhixing.nlp.NlpGbtApp"
JAR="/home/webapps/ppai-algorithm/nlp/target/scala-2.11/nlp_2.11-1.0.jar"

EXECUTOR_MEMORY="100G"
EXECUTOR_CORES=8


function build() {
    ${SBT_BIN} package
    if [ ! $? -eq 0 ]; then
        exit 1
    fi
}


function run() {
    nohup ${SPARK_SUBMIT} \
        --master local[2] \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${EXECUTOR_MEMORY} \
        --driver-cores ${EXECUTOR_CORES} \
        --class ${CLASS} \
        --conf spark.memory.fraction="0.8" \
        --conf spark.kryoserializer.buffer.max="1800m" \
        ${JAR} &
    spin $!
}


function run_gbt() {
    nohup ${SPARK_SUBMIT} \
        --master local[*] \
        --executor-memory ${EXECUTOR_MEMORY} \
        --driver-memory ${EXECUTOR_MEMORY} \
        --driver-cores ${EXECUTOR_CORES} \
        --class ${GBT_CLASS} \
        --conf spark.memory.fraction="0.8" \
        --conf spark.kryoserializer.buffer.max="1800m" \
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
run_gbt)
    build
    run_gbt
;;
generate)
    python generate_submit_file.py $2
;;
*)
    echo "Usage: ./test.sh [cmd]"
    echo "    cmd:"
    echo "        build"
    echo "        run"
    echo "        run_gbt"
    echo "        generate"
;;
esac
