#!/bin/sh
# Please modify OPENAI_API_BASE, OPENAI_API_KEY in this script and run script.

set -e

export OPENAI_API_BASE=
export OPENAI_API_KEY=

MODEL_NAME=$1

sh scripts/0_run_indiv.sh $MODEL_NAME
sh scripts/1_run_postprocess_indiv.sh $MODEL_NAME
sh scripts/2_score_indiv.sh $MODEL_NAME
sh scripts/3_run_rims.sh $MODEL_NAME
sh scripts/3_run_sg.sh $MODEL_NAME
sh scripts/4_postprocess_and_score_rims.sh $MODEL_NAME
sh scripts/4_postprocess_and_score_sg.sh $MODEL_NAME