# MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning

## Setup

We implement MoRA in peft-mora based on HF peft in the [`apply_mora`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L229) and [`get_delta_weight`](https://github.com/kongds/MoRA/blob/main/peft-mora/src/peft/tuners/lora/layer.py#L514).
``` sh
pip install -e ./peft-mora
```

After installation, it can be used like

``` python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    use_mora=True, # enable mora
    mora_type=1, # type 1 refer to Eq. 6, type 6 (RoPE based) for small ranks refer to Eq. 9 in paper.
    r=lora_r, # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
    **kwargs,
)
model = get_peft_model(model, config)

# training here...

model = model.merge_and_unload() # can be merged into model via `merge_and_unload` like LoRA
```

## Examples
### fine-tuning MetaMath with MoRA

``` sh
RANK=8
deepspeed --num_gpus=8 --num_nodes=2 train.py \
           --base_model <LLAMA-2> --micro_batch_size 4\
            --wandb_run_name mora_math_r8 --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
            --num_epochs 3 --deepspeed ds.config --wandb_project lora-math --lora_r $RANK --batch_size 128 \
            --data_path meta-math/MetaMath \
            --save_steps 3000 \
            --learning_rate 3e-4 --mora_type 6 \
            --logging_steps 5  --use_bf16  --use_16bit --use_mora 
```

### pretraining

``` sh
deepspeed --num_gpus=8 --num_nodes=4 train.py \
        --micro_batch_size 16 --wandb_run_name mora-pretrain250m-r128 \
        --num_epochs 1 --wandb_project lora-pretrain --batch_size 1024 \
        --data_path <processed C4> --logging_steps 1 \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
        --lora_r 128 --lora_alpha 64 --warmup_steps 1000  \
        --force_tqdm_update --lr_scheduler_type cosine \
        --max_steps 10000 --pretrain 250m \
        --train_embhead --learning_rate 5e-4 \
        --use_mora --use_relora --use_relora_step 2000  # ReMoRA merge per 2000 steps 
```

## Acknowledgement
Our Code is based on peft, alpaca-lora and ReLoRA
