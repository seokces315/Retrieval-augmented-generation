#! /bin/bash

scripts="../src/main.py"

learning_rate=5e-5
weight_decay=0.0
max_grad_norm=1.0
warmup_ratio=0.0

python3 $scripts \
	--learning_rate $learning_rate \
	--weight_decay $weight_decay \
	--max_grad_norm $max_grad_norm \
	--warmup_ratio $warmup_ratio
