REPO_PATH=$(git rev-parse --show-toplevel)

train_file="${REPO_PATH}/data/subtask1-document/en-train.json"
test_file=$train_file
model="roberta-base"
dest="${REPO_PATH}/models/roberta-subtask1"
python transformer_models.py --train_file  $train_file --test_file $test_file --dest $dest --model $model
