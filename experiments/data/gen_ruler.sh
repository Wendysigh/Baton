#!/bin/bash
# RULER data. Clone NVIDIA/RULER into $BATON_CACHE/ruler_repo/RULER at commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a and
# apply ruler_vt_hard.diff (adds the 3-chain variable-tracking config) before running this.
# Test: seeds 20261103/4/5 (167/167/166 per task and length). Development: seed 7777, 8 per task and length.
R=$BATON_CACHE/ruler_repo/RULER/scripts/data; D=$BATON_CACHE/rev_ruler; PY=${PYTHON:-python}
gen() { # seed n
  for L in 4096 16384; do for T in vt vt_hard fwe cwe niah_single_1 niah_multikey_2 niah_multikey_3; do
    [ -f $D/s$1/len$L/$T/validation.jsonl ] && [ $(wc -l < $D/s$1/len$L/$T/validation.jsonl) -ge $2 ] && continue
    ( cd $R && $PY prepare.py --save_dir $D/s$1/len$L --benchmark synthetic --task $T --tokenizer_path $BATON_MODELS/Qwen3.5-9B --tokenizer_type hf --max_seq_length $L --model_template_type base --num_samples $2 --random_seed $1 > /dev/null 2>&1 || echo FAIL $1 $L $T ) &
  done; wait; done
}
gen 7777 8; gen 20261103 167; gen 20261104 167; gen 20261105 166; echo gen done
