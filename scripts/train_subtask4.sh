train_file="./data/subtask4-token/en-train.txt"
test_file=$train_file
model="roberta-base"
dest="./models/roberta-subtask4"
python transformer_models_BIO.py --train_file  $train_file --test_file $test_file --dest $dest --model $model