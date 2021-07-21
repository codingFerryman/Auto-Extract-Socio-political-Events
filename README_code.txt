code is rather simple, it takes by default some required arguments, e.g.

--train_file (path to train_file)
--test_file (path to test_file, for development purposes, this would be the same as train_file and we split into train_dev in the script if args.train_file == args.test_file)
--model (the model which we are using, e.g. roberta-base)


and then, we just train a standard transformer models.
because I trained some of the bigger models, e.g. debertav2-xl-mnli or bigbird-large, I used

--gradient_checkpointing
--gradient_accumulation
--FP16 training

all neat tricks to reduce memory usage, they're all called by default in the scripts.

to train subtask 1 model, submit the following job


train_file="../20210312/subtask1-document/en-train.json"
test_file="../20210312/subtask1-document/en-train.json"
model="/cluster/work/lawecon/Work/dominik/roberta-base" # or just "roberta-base"
dest="dest roberta-task1" # directory to save models
bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" python transformer_models.py --train_file  $train_file --test_file $test_file --dest $dest --model $model

# predict subtask 1 on test data
train_file="../20210312/subtask1-document/en-train.json"
test_file="test_subtask1.json"
model="dest roberta-task1"
dest="dest roberta-task1"

bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" python transformer_models.py --train_file  $train_file --test_file $test_file --dest $dest --model $model --only_prediction "true"

all the other subtasks work equivalenty, for subtask 2, we just have to change train and test files

for subtask 4, we can use transformer_models_BIO.py which has the same logic, but implements token-level prediction instead of sequence-level predictions



