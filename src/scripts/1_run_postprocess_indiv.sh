# process_indiv
MODEL=$1
F1=outputs/gsm8K_test_dt.gsm/$MODEL/n1_baseline_raw_query_result.jsonl
F2=outputs/ocw_course_dt.ocw/$MODEL/n1_baseline_raw_query_result.jsonl
F3=outputs/MATH-full_dt.math/$MODEL/n1_baseline_raw_query_result.jsonl
F4=outputs/SVAMP_dt.svamp/$MODEL/n1_baseline_raw_query_result.jsonl

for F in $F1 $F2 $F3 $F4; do
    python postprocess_rawouts.py process_indiv --infile $F
done
