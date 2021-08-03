import json
import random
import argparse
import sys

from sklearn.metrics import precision_recall_fscore_support, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, BigBirdConfig
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2ForSequenceClassification, DebertaV2Config


class SequenceClassificationDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.examples = list(zip(x, y))
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        model_inputs = self.tokenizer([i[0] for i in batch], return_tensors="pt", padding=True, truncation=True,
                                      max_length=64).to(self.device)
        labels = torch.tensor([i[1] for i in batch]).to(self.device)
        return {"model_inputs": model_inputs, "label": labels}


class RandomModel():
    def __init__(self):
        pass

    def fit(self, data):
        """
        Learns the seed for future prediction.
        Doesn't use the given data.
        """
        self.seed = random.choice(range(100))

    def predict(self, test_data):
        """
        Takes some data and makes predictions based on the seed which was learnt in the fit() part.
        Returns the predictions.
        """
        random.seed(self.seed)
        preds = [{"id": instance['id'], "prediction": random.choice([0, 1])} for instance in test_data]
        return preds


def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data


def evaluate_epoch(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)):
            output = model(**batch["model_inputs"])
            logits = output.logits
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(targets, outputs, labels=[0, 1],
                                                                                 average="macro")
    print(precision_macro, recall_macro)
    print("F1-macro score for test data predictions are: %.4f" % f1_macro)
    return outputs


def evaluate_old(goldfile, sysfile):
    """
    Takes goldfile (json) and sysfile (json) paths.
    Prints out the results on the terminal.
    The metric used is F1-Macro implementation from sklearn library (Its documentation is at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

    This function is the exact way the subtask1's submissions will be evaluated.
    """
    gold = {i["id"]: i["label"] for i in read(goldfile)}
    sys = {i["id"]: i["prediction"] for i in read(sysfile)}

    labels, preds = [], []
    for idx in gold:
        labels.append(gold[idx])
        preds.append(sys[idx])

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1],
                                                                                 average="macro")
    print("F1-macro score for test data predictions are: %.4f" % f1_macro)


def evaluate(gold, predictions):
    """
    Takes goldfile (json) and sysfile (json) paths.
    Prints out the results on the terminal.
    The metric used is F1-Macro implementation from sklearn library (Its documentation is at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

    This function is the exact way the subtask1's submissions will be evaluated.
    """
    # gold = {i["id"]:i["label"] for i in read(goldfile)}
    # sys = {i["id"]:i["prediction"] for i in read(sysfile)}

    labels, preds = [], []
    for idx in gold:
        labels.append(gold[idx])
        preds.append(predictions[idx])

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1],
                                                                                 average="macro")
    print("F1-macro score for test data predictions are: %.4f" % f1_macro)
    return preds


def main(train_file, test_file=None):
    train_data = read(train_file)
    try:
        X = [i["text"] for i in train_data]
    except:
        X = [i["sentence"] for i in train_data]

    y = [i["label"] for i in train_data]
    idx = [i["id"] for i in train_data]
    # if we only provide train file, we do train-test split

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size=0.33, random_state=42)

    if test_file != train_file:
        # else, we are in inference mode and predict the testset
        test_data = read(test_file)
        try:
            X_test = [i["text"] for i in test_data]
        except:
            X_test = [i["sentence"] for i in test_data]

        y_test = [0] * len(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "/cluster/work/lawecon/Work/dominik/roberta-base"
    model_name = "/cluster/work/lawecon/Work/dominik/FEVER_bigbird/bigbird-roberta-base"
    # model_name = "roberta-base"
    model_name = args.model

    if "bigbird" in model_name:
        config = BigBirdConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True
        model = BigBirdForSequenceClassification.from_pretrained(model_name, config=config).to(device)
        tokenizer = BigBirdTokenizer.from_pretrained(model_name)

    elif "roberta" in model_name:
        model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

    elif "deberta" in model_name:
        # DebertaV2Tokenizer, DebertaV2Model, DebertaV2ForSequenceClassification, DebertaV2Config
        config = DebertaV2Config.from_pretrained(model_name)
        config.gradient_checkpointing = True
        config.num_classes = 2
        model = DebertaV2ForSequenceClassification.from_pretrained(model_name, config=config).to(device)
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    trainset = SequenceClassificationDataset(X_train, y_train, tokenizer)
    devset = SequenceClassificationDataset(X_test, y_test, tokenizer)

    warmup_steps = 0
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate_fn)
    t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    model.zero_grad()
    optimizer.zero_grad()

    cuda_device_capability = torch.cuda.get_device_capability()
    if cuda_device_capability[0] >= 8:
        use_amp = True
    else:
        use_amp = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.only_prediction is not None:
        preds = evaluate_epoch(model, devset)
        save_path = os.path.join(args.dest)
        with open(os.path.join(save_path, "dev_preds.txt"), "w") as f:
            for i in preds:
                f.write(str(i) + "\n")
        sys.exit(0)

    for epoch in range(args.num_epochs):
        model.train()
        t = tqdm(train_dataloader)
        for i, batch in enumerate(t):
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch["model_inputs"], labels=batch['label'])
                loss = output.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
            acc = (output.logits.argmax(axis=-1).detach() == batch["label"]).float().mean()
            t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}, acc: {round(acc.item(), 4)}')

        preds = evaluate_epoch(model, devset)
        # Save
        save_path = os.path.join(args.dest)
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
    with open(os.path.join(save_path, "dev_preds.txt"), "w") as f:
        for i in preds:
            f.write(str(i) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="The path to the training data json file")
    parser.add_argument('-test_file', '--test_file', required=True, help="The path to the training data json file")
    parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
    parser.add_argument('--model', type=str, default='roberta-large')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--only_prediction", default=None, type=str,
                        help="Epsilon for Adam optimizer.")

    args = parser.parse_args()

    main(args.train_file, args.test_file)

# subtask 1 roberta, just change model names if we use different model
# bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask1-document/en-train.json --test_file ../20210312/subtask1-document/en-train.json --dest roberta-task1 --model /cluster/work/lawecon/Work/dominik/roberta-base

# evaluate

# subtask 2 roberta
# bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask2-sentence/en-train.json --test_file ../20210312/subtask2-sentence/en-train.json --dest roberta-task2 --model /cluster/work/lawecon/Work/dominik/roberta-base

# subtask 2 deberta

# bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask2-sentence/en-train.json --test_file ../20210312/subtask2-sentence/en-train.json --dest debertav2-task2 --model ../../../deberta-v2-xlarge-mnli/ --batch_size 1 --gradient_accumulation_steps 16


# bigbird-roberta-large
# bsub -n 1 -R "rusage[mem=25600,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask1-document/en-train.json --test_file ../20210312/subtask1-document/en-train.json --dest bigbird-large-task1 --gradient_accumulation_steps 16 --batch_size 2 --model /cluster/work/lawecon/Work/dominik/FEVER_bigbird/bigbird-roberta-large

# evaluate
# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask1-document/en-train.json --test_file ../test.json --dest bigbird-large-task1 --batch_size 2 --model bigbird-large-task1 

# evaluate subtask 2
# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python task1/subtask1/transformer_models.py --train_file ../20210312/subtask1-document/en-train.json --test_file ../test_subtask2.json --dest debertav2-task2 --batch_size 2 --model debertav2-task2
