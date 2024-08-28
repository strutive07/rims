MODEL_NAME=$1

python run_baseline.py --gsm_jslf ../dataset/gsm8K_test.jsonl --dataset_type gsm --backbone $MODEL_NAME
python run_baseline.py --gsm_jslf ../dataset/ocw/ocw_course.jsonl --dataset_type ocw --backbone $MODEL_NAME
python run_baseline.py --gsm_jslf ../dataset/MATH/MATH-full.jsonl --dataset_type math --backbone $MODEL_NAME
python run_baseline.py --gsm_jslf ../dataset/SVAMP/SVAMP.json --dataset_type svamp --backbone $MODEL_NAME