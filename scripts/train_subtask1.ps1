$REPO_PATH=(git rev-parse --show-toplevel)

$train_file=(JOIN-PATH $REPO_PATH "data/subtask1-document/en-train.json")
$test_file=$train_file
$model="roberta-base"
$dest=(JOIN-PATH $REPO_PATH "models/roberta-subtask1")
python (JOIN-PATH $REPO_PATH "src/subtask_1_2/transformer_models.py") --train_file  $train_file --test_file $test_file --dest $dest --model $model
