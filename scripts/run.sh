#!/bin/bash

# Loading the required module
source /etc/profile

# Load input
CONF=$1
SDGYM=${2:-$(which sdgym)}
BASE_PATH=${3:-$(pwd)}
PRIVBAYES_BIN=$4
DATASETS_PATH=$5
ITERATIONS=$6
RUN_ID=$7

# Source config
. $CONF

if [ -n "$PRIVBAYES_BIN" ]; then
    export PRIVBAYES_BIN
fi

if [ -n "$RUN_ID" ]; then
    NAME=$NAME-$RUN_ID
    export RUN_ID
fi

# Define paths
if [ "x$CACHE" != "xFALSE" ]; then
    CACHE_PATH="$BASE_PATH/cache/$NAME"
    mkdir -p $CACHE_PATH
fi

LOG_PATH=$BASE_PATH/logs/$NAME.log
OUTPUT_PATH=$BASE_PATH/output/$NAME.csv

# Enter base path
cd $BASE_PATH

# Ensure dirs exist
mkdir -p logs
mkdir -p output

# Run the script
$SDGYM run -v \
    -o $OUTPUT_PATH \
    -l $LOG_PATH \
    -t ${TIMEOUT:-28800} \
    -i ${ITERATIONS:-1} \
    ${MODALITIES:+-dm ${MODALITIES}} \
    ${CACHE_PATH:+-c ${CACHE_PATH}} \
    ${DATASETS_PATH:+-dp ${DATASETS_PATH}} \
    -W ${GPU:-${CPU}} \
    -s $SYNTHESIZERS
