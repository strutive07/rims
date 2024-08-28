MODEL=$1

F1=outputs/gsm8K_test_dt.gsm/$MODEL/simple_greedy/n1_0.0_sg_raw_query_result.jsonl
F2=outputs/ocw_course_dt.ocw/$MODEL/simple_greedy/n1_0.0_sg_raw_query_result.jsonl
F3=outputs/MATH-full_dt.math/$MODEL/simple_greedy/n1_0.0_sg_raw_query_result.jsonl
F4=outputs/SVAMP_dt.svamp/$MODEL/simple_greedy/n1_0.0_sg_raw_query_result.jsonl

for F in $F1 $F2 $F3 $F4; do
    python postprocess_rawouts.py process_simple_greedy --infile $F
    python postprocess_rawouts.py process_simple_greedy --infile $F
done


F1=outputs/gsm8K_test_dt.gsm/$MODEL/simple_greedy/processed_sg.jsonl
F2=outputs/ocw_course_dt.ocw/$MODEL/simple_greedy/processed_sg.jsonl
F3=outputs/MATH-full_dt.math/$MODEL/simple_greedy/processed_sg.jsonl
F4=outputs/SVAMP_dt.svamp/$MODEL/simple_greedy/processed_sg.jsonl

for F in $F1 $F2 $F3 $F4; do
    python score_processed.py score_sg --ptn $F
done