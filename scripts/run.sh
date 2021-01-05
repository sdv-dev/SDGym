#!/bin/bash

# Loading the required module
source /etc/profile

# Load input
CONF=$1
SDGYM=${2:-$(which sdgym)}
BASE_PATH=${3:-$(pwd)}
PRIVBAYES_BIN=$4

if [ -n "$PRIVBAYES_BIN" ]; then
    export PRIVBAYES_BIN
fi

# Source config
. $CONF

# Set default values
WORKERS=${GPU:-${CPU}}
TIMEOUT=${TIMEOUT:-28800}
ITERATIONS=${ITERATIONS:-3}
MODALITIES=${MODALITIES:+-dm ${MODALITIES}}

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
    -t $TIMEOUT \
    -i 3 \
    $MODALITIES \
    ${CACHE_PATH:+-c ${CACHE_PATH}} \
    -W $WORKERS \
    -s $SYNTHESIZERS
