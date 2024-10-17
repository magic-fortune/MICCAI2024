python fixmatch.py \
--exp "fixmatch/model" \
--conf_thresh 0.75 \
--label_num 30 \
--max_iterations 20000 \
--optimizer AdamW \
--base_lr 0.0005 \
--eta 0.75 \
--s1_to_s2 \
--deterministic 0 \
--batch_size 2 \
--corr_match_type kl \
--temperature 1.25

