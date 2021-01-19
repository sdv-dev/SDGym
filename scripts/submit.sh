#!/bin/bash


function submit() {
    CONF=$(realpath $CONF)
    . $CONF

    # Set GPU input
    GPU=${GPU:+-g volta:${GPU}}

    if [ "${PARALLEL:-$PARALLEL_ARG}" == "TRUE" ]; then
        RUN_ID=0
        RAW_NAME=$NAME
        for i in $(seq 1 "$ITERATIONS"); do
            NAME=$RAW_NAME-$RUN_ID
            LLsub run.sh -s $CPU $GPU -o $BASE_PATH/jobs/$NAME.log -J $NAME -- \
                $CONF $SDGYM $BASE_PATH $PRIVBAYES_BIN $DATASETS_PATH 1 $RUN_ID
            let "RUN_ID+=1"
        done
    else
        LLsub run.sh -s $CPU $GPU -o $BASE_PATH/jobs/$NAME.log -J $NAME -- \
            $CONF $SDGYM $BASE_PATH $PRIVBAYES_BIN $DATASETS_PATH $ITERATIONS
    fi
}

function fail() {
    echo ERROR: $* 1>&2
    echo "Usage: $0 [-p] [-i ITERATIONS] CONFIG [CONFIG...]"
    exit 1
}


while getopts ":i:p" opt; do
    case ${opt} in
        p )
            PARALLEL_ARG=TRUE
            ;;
        i )
            ITERATIONS=$OPTARG
            [[ "$ITERATIONS" -gt 0 ]] || fail "ITERATIONS must be a positive integer"
            ;;
        \? )
            fail "Invalid option: $OPTARG"
            ;;
        : )
            fail "Invalid option: $OPTARG requires an argument"
            ;;
    esac
done

shift $((OPTIND -1))

ITERATIONS=${ITERATIONS:-3}

if [ $# -eq 0 ]; then
    fail "No configs provided"
fi

SDGYM=$(which sdgym)

if [ -z "$SDGYM" ]; then
    fail "sdgym command not found"
fi

DATASETS_PATH=$(pwd)/datasets
if [ ! -d $DATASETS_PATH ]; then
    sdgym download-datasets -dp $DATASETS_PATH -v
fi

PRIVBAYES_BIN=$(realpath ../privbayes/privBayes.bin)

BASE_PATH=$(pwd)/runs/$(date +%Y-%m-%dT%H%M%S)
mkdir -p $BASE_PATH/jobs

for CONF in $*; do
    submit
done
