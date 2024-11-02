CUDA_VISIBLE_DEVICES=0 python experiments.py --task transfer --run_type select  --organ Renal-Transparency

CUDA_VISIBLE_DEVICES=0 python experiments.py --task transfer --run_type all  --organ Renal-Transparency

CUDA_VISIBLE_DEVICES=0 python experiments.py --task further --run_type select  --organ Renal-Transparency

CUDA_VISIBLE_DEVICES=0 python experiments.py --task further --run_type all  --organ Renal-Transparency

CUDA_VISIBLE_DEVICES=0 python experiments.py --task time --run_type select  --organ Renal-Transparency

CUDA_VISIBLE_DEVICES=0 python experiments.py --task time --run_type all  --organ Renal-Transparency

