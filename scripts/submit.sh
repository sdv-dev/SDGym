#!/bin/bash


function submit() {
    # Load config
    CONF=$(realpath $1)
    SDGYM=$2
    BASE_PATH=$3
    PRIVBAYES_BIN=$4

    . $CONF

    # Set GPU input
    GPU=${GPU:+-g volta:${GPU}}

    # Submit
    LLsub run.sh -s $CPU $GPU -o $BASE_PATH/jobs/$NAME.log -J $NAME -- $CONF $SDGYM $BASE_PATH $PRIVBAYES_BIN
}

SDGYM=$(which sdgym)

if [ -z "$SDGYM" ]; then
    echo "ERROR: sdgym command not found"
    exit 1
fi

PRIVBAYES_BIN=$(realpath ../privbayes/privBayes.bin)

BASE_PATH=$(pwd)/runs/$(date +%Y-%m-%dT%H%M%S)
mkdir -p $BASE_PATH/jobs

exit

for CONF in $*; do
    submit $CONF $SDGYM $BASE_PATH $PRIVBAYES_BIN
done
