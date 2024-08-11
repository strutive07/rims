set -x
MODEL=$1

PTN=outputs/*/$MODEL/rims/rims_*/
python postprocess_rawouts.py process_rims --ptn "$PTN" --n 1

PTN1=outputs/*/$MODEL/rims/rims*/processed_rims.jsonl
python score_processed.py score_rims --ptn "$PTN1" --n 1
