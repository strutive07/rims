GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1

# mkdir originals/
for F in $GSM_RIMS_RW $GSM_RIMS_RW_1 $GSM_RIMS_RW_2 $OCW_RIMS $OCW_RIMS_1 $MATH_RIMS $MATH_RIMS_1; do
    cp ../../../$F originals/
done

cp originals/$(basename "$GSM_RIMS_RW") ./rims_gsm0.txt
cp originals/$(basename "$GSM_RIMS_RW_1") ./rims_gsm1.txt
cp originals/$(basename "$GSM_RIMS_RW_2") ./rims_gsm2.txt
cp originals/$(basename "$OCW_RIMS") ./rims_ocw0.txt
cp originals/$(basename "$OCW_RIMS_1") ./rims_ocw1.txt
cp originals/$(basename "$MATH_RIMS") ./rims_math0.txt
cp originals/$(basename "$MATH_RIMS_1") ./rims_math1.txt
