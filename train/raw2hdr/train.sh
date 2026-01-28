export MODEL_DIR="../../models/Flux"
export OUTPUT_DIR="output"
export CONFIG="config.yaml"
export TRAIN_DATA="examples/train.jsonl" # your data jsonl file

accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --input_is_raw \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --seed 42 \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=8 \
    --validation_steps=250 \
    --checkpointing_steps=2000 \
    --subject_test_images None \
    --cond_width 512 \
    --cond_height 512 \
    --test_w 512 \
    --test_h 512 \
    --num_validation_images=1 \
    --validation_prompt " " \
    --spatial_test_images "examples/test_raw.exr"