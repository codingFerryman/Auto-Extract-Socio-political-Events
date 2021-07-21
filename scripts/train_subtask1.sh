train_file="./data/subtask1-document/en-train.json"
test_file=$train_file
model="roberta-base"
dest="./models/roberta-subtask1"
python transformer_models.py --train_file  $train_file --test_file $test_file --dest $dest --model $model