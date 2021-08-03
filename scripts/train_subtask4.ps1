$env:CUDA_VISIBLE_DEVICES="1"

$REPO_PATH=(git rev-parse --show-toplevel)

$train_file=(JOIN-PATH $REPO_PATH "data/subtask4-token/en-train.txt")
$test_file=$train_file
$model="google/bigbird-roberta-base"
$dest=(JOIN-PATH $REPO_PATH "models/bigbird-subtask4")
$gradient_accumulation_steps=4
python (JOIN-PATH $REPO_PATH "src/subtask_4/transformer_models_BIO.py") --train_file  $train_file --test_file $test_file --dest $dest --model $model --gradient_accumulation_steps $gradient_accumulation_steps
