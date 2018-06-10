#!/bin/sh



function debug() {
    echo "`date +"%Y-%m-%d %H:%M:%S"` DEBUG:: $*"
}

function warning() {
    echo "`date +"%Y-%m-%d %H:%M:%S"` WARNI:: $*"
}

function error() {
    echo "`date +"%Y-%m-%d %H:%M:%S"` ERROR:: $*"
}

function spin() {
    pid=$1

    offset=0
    interval="0.1"
    sp='/-\|'
    start=`date +%s`

    printf ' '

    while kill -0 $pid 2>/dev/null
    do
        timeName=""

        if [ $offset -gt 360000 ]; then
            hours=`echo "${offset} / 600 / 600" | bc -l`
            timeName=`printf '%.1fh' "$hours"`
        elif [ $offset -gt 600 ] ; then
            minutes=`echo "${offset} / 600" | bc -l`
            timeName=`printf '%.1fm' "$minutes"`
        else
            seconds=`echo "${offset} / 10.0" | bc -l`
            timeName=`printf '%.1fs' "$seconds"`
        fi

        printf '\b\b\b\b\b\b\b\b\b\b%03.1s %s' "$sp" "$timeName"
        sp=${sp#?}${sp%???}

        sleep ${interval}
        offset=`expr $offset + 1`
    done

    end=`date +%s`

    wait $pid
    exitCode=$?
    echo ""
    debug "Pid ${pid} exit code ${exitCode}.(`expr $end - $start`s)"
}

