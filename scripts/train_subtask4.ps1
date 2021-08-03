$REPO_PATH=(git rev-parse --show-toplevel)

$train_file=(JOIN-PATH $REPO_PATH "data/subtask4-token/en-train.txt")
$test_file=$train_file
$model="roberta-base"
$dest=(JOIN-PATH $REPO_PATH "models/roberta-subtask4")
python (JOIN-PATH $REPO_PATH "src/subtask_4/transformer_models_BIO.py") --train_file  $train_file --test_file $test_file --dest $dest --model $model
