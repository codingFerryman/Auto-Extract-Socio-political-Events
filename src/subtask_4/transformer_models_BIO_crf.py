import json
import random
import argparse
from sklearn.metrics import precision_recall_fscore_support, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaConfig
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, BigBirdConfig, BigBirdForTokenClassification, BigBirdModel
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from conlleval import evaluate_conll_file

from transformers import DebertaV2Tokenizer, DebertaV2Model, DebertaV2ForSequenceClassification, DebertaV2Config


import torch
from TorchCRF import CRF


class SequenceClassificationDataset(Dataset):
	def __init__(self, examples, tokenizer):
		self.examples = examples
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.labels = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O", "PAD"]
		self.label2id = {j:i for i,j in enumerate(self.labels)}
		self.id2label = {i:j for i,j in enumerate(self.labels)}

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = {"input_ids":[]}
		batch_labels = []
		index2labels = []
		for sent, labels in batch:
			encoded_sent, encoded_labels = [self.tokenizer.cls_token_id], [self.label2id["O"]]
			index2label = {}
			for i, (token, label) in enumerate(zip(sent, labels)):
				sub_tokens = self.tokenizer.encode(token, add_special_tokens=False)
				sub_labels = [self.label2id[label]] * len(sub_tokens)
				index2label[i] = len(encoded_labels)
				encoded_sent.extend(sub_tokens)
				encoded_labels.extend(sub_labels)
			encoded_sent.append(self.tokenizer.sep_token_id)
			encoded_labels.append(self.label2id["O"])
			model_inputs["input_ids"].append(encoded_sent)
			batch_labels.append(encoded_labels)
			index2labels.append(index2label)
		# pad to max length
		max_len = max(len(i) for i in model_inputs["input_ids"])
		#if max_len > 512:
		#	print (batch)
		#	sys.exit(0)
		model_inputs["attention_mask"] = []
		for i,j in zip(model_inputs["input_ids"], batch_labels):
			#print (j)
			length = len(i)
			# pad
			i.extend([self.tokenizer.pad_token_id] * (max_len - length))
			model_inputs["attention_mask"].append([1] * length + [0] * (max_len - length))
			j.extend([self.label2id["PAD"]] * (max_len - length))
		model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"]).to(self.device)
		model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"]).to(self.device)
		batch_labels = torch.tensor(batch_labels).to(self.device)
		assert model_inputs["input_ids"].shape == model_inputs["attention_mask"].shape == batch_labels.shape

		#print (model_inputs["input_ids"].shape, model_inputs["attention_mask"].shape, batch_labels.shape)

		return {"model_inputs": model_inputs, "label": batch_labels}, index2labels
			
class RandomModel():
	def __init__(self):
		pass

	def fit(self,data):
		"""
		Learns the seed for future prediction.
		Doesn't use the given data.
		"""
		self.seed = random.choice(range(100))


	def predict(self,test_data):
		"""
		Takes some data and makes predictions based on the seed which was learnt in the fit() part.
		Returns the predictions.
		"""
		random.seed(self.seed)
		preds = [{"id":instance['id'], "prediction":random.choice([0,1])} for instance in test_data]
		return preds

def read(path):
	"""
	Reads the file from the given path (json file).
	Returns list of instance dictionaries.
	"""
	data = []
	with open(path, "r", encoding="utf-8") as f:
		sent, labels = [], []
		for line in f:
			line = line.strip()
			if not line:
				data.append((sent, labels))
				sent, labels = [], []
				continue
			line = line.split()

			if len(line) == 2:			
				sent.append(line[0])
				labels.append(line[1])
			elif len(line) == 1:
				sent.append(line[0])
				labels.append("O")

	print (data[0])
	print (len(data))
	return data

def evaluate_epoch(model, dataset, crf):
	model.eval()
	targets = []
	outputs = []
	words = []

	labels = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O", "PAD"]
	label2id = {j:i for i,j in enumerate(labels)}
	id2label = {i:j for i,j in enumerate(labels)}

	with torch.no_grad():
		for (batch, index2labels), (tokens, true_labels) in tqdm(zip(DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn), dataset.examples)):
			#print (tokens, true_labels)
			#output = model(**batch["model_inputs"])
			logits = forward_crf(crf, batch, model, mode="test")[0]
			#logits = output.logits[0]
			#print (logits)
			indices_labels = [index2labels[0][i] for i in range(len(true_labels))]
			#print (indices_labels)
			#print (logits.shape)
			predicted_labels = [id2label[logits[i]] for i in indices_labels]
			#for i,j,k in zip(tokens, true_labels, predicted_labels):
			#	print (i,j,k)
			targets.append(true_labels)
			outputs.append(predicted_labels)
			words.append(tokens)
			#if len(targets) > 2: 
			#	break
			#targets.extend(batch['label'].float().tolist())
			#outputs.extend(logits.argmax(dim=1).tolist())

	lines = []
	with open("subtask4_predictions.tsv", "w") as outfile:
		for i_,j_,k_ in zip(words, targets, outputs):
			for i,j,k in zip(i_, j_, k_):
				lines.append(" ".join((i,j,k)))
				outfile.write(" ".join((i,j,k)) + "\n")
			lines.append("")
			outfile.write("\n")
	evaluate_conll_file(lines)


def evaluate_old(goldfile, sysfile):
	"""
	Takes goldfile (json) and sysfile (json) paths.
	Prints out the results on the terminal.
	The metric used is F1-Macro implementation from sklearn library (Its documentation is at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

	This function is the exact way the subtask1's submissions will be evaluated.
	"""
	gold = {i["id"]:i["label"] for i in read(goldfile)}
	sys = {i["id"]:i["prediction"] for i in read(sysfile)}

	labels, preds = [], []
	for idx in gold:
		labels.append(gold[idx])
		preds.append(sys[idx])

	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, labels=[0,1], average="macro")
	print("F1-macro score for test data predictions are: %.4f" %f1_macro)

def evaluate(gold, predictions):
	"""
	Takes goldfile (json) and sysfile (json) paths.
	Prints out the results on the terminal.
	The metric used is F1-Macro implementation from sklearn library (Its documentation is at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

	This function is the exact way the subtask1's submissions will be evaluated.
	"""
	#gold = {i["id"]:i["label"] for i in read(goldfile)}
	#sys = {i["id"]:i["prediction"] for i in read(sysfile)}

	labels, preds = [], []
	for idx in gold:
		labels.append(gold[idx])
		preds.append(predictions[idx])

	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, labels=[0,1], average="macro")
	print("F1-macro score for test data predictions are: %.4f" %f1_macro)
	return preds


def forward_crf(crf, batch, model, mode="train"):
	hidden = model(**batch["model_inputs"]).logits
	if mode == "train":
		crf_out = crf.forward(hidden, batch["label"], batch["model_inputs"]["attention_mask"].byte())
		#print (crf_out.shape)
		crf_out = crf_out.mean() * -1
	else:
		crf_out = crf.viterbi_decode(hidden, batch["model_inputs"]["attention_mask"].byte()) #list batchsize + one prediction for each label
	return crf_out

def main(train_file, test_file=None):

	# Create model.
	# model = RandomModel()
	# Read training data.
	train_data = read(train_file)
	random.shuffle(train_data)
	X_train, X_test = train_data[:600], train_data[600:]

	if test_file != train_file:
		X_test = read(test_file)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model_name = args.model

	num_labels = 16
	crf = CRF(num_labels).to(device)

	# damn, should decide on sequence lenght, what to do?
	# he, don't need that, this is transformer output!
	#hidden = torch.randn((args.batch_size, sequence_size, num_labels), requires_grad=True).to(device)


	"""
	mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device) # (batch_size. sequence_size)
	labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
	hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True).to(device)
	crf = CRF(num_labels)

	Computing log-likelihood (used where forward)

	crf.forward(hidden, labels, mask)
	tensor([-7.6204, -3.6124], device='cuda:0', grad_fn=<ThSubBackward>)

	Decoding (predict labels of sequences)

	crf.viterbi_decode(hidden, mask)
	[[0, 2, 2], [4, 0]]
	"""

	config = BigBirdConfig.from_pretrained(model_name)
	config.gradient_checkpointing = True
	config.num_labels = 16
	model = BigBirdForTokenClassification.from_pretrained(model_name, config=config).to(device)
	tokenizer = BigBirdTokenizer.from_pretrained(model_name)


	trainset = SequenceClassificationDataset(X_train, tokenizer)
	devset = SequenceClassificationDataset(X_test, tokenizer)

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

	use_amp = True
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	# only evaluate
	if args.only_prediction is not None:
		preds = evaluate_epoch(model, devset, crf)
		sys.exit(0)

	for epoch in range(args.num_epochs):
		model.train()
		t = tqdm(train_dataloader)
		for i, (batch, _) in enumerate(t):
			#if i > 2:
			#	break

			"""
			output = model(**batch["model_inputs"], labels=batch['label'])
			loss = output.loss / args.gradient_accumulation_steps
			loss.backward()
			if (i + 1) % (args.gradient_accumulation_steps) == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
				t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
			"""

			with torch.cuda.amp.autocast(enabled=use_amp):
				#output = model(**batch["model_inputs"], labels=batch['label'])
				#loss = output.loss / args.gradient_accumulation_steps
				#def forward_crf(crf, batch, model, mode="train"):

				loss = forward_crf(crf, batch, model).to(device)
			scaler.scale(loss).backward()

			if (i + 1) % args.gradient_accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()  # Update learning rate schedule
				optimizer.zero_grad()
			#acc = (output.logits.argmax(axis=-1).detach() == batch["label"]).float().mean()
			t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')

		preds = evaluate_epoch(model, devset, crf)
		# Save
		save_path = os.path.join(args.dest)
		os.makedirs(save_path, exist_ok=True)
		tokenizer.save_pretrained(save_path)
		model.save_pretrained(save_path)
	"""
	with open(os.path.join(save_path, "dev_preds.txt"), "w") as f:
		for i in preds:
			f.write(str(i) + "\n")
	"""

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

# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" PYTHONPATH="." python task1/subtask1/transformer_models_BIO_crf.py --train_file ../20210312/subtask4-token/en-train.txt --test_file ../20210312/subtask4-token/en-train.txt --dest subtask-4-bigbird-base-crf --model ../../../FEVER_bigbird/bigbird-roberta-base/ --gradient_accumulation_steps 4 --only_prediction yes

# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -R volta PYTHONPATH="." python task1/subtask1/transformer_models_BIO_crf.py --train_file ../20210312/subtask4-token/en-train.txt --test_file test_subtask4.txt --dest subtask-4-bigbird-large-crf --model bigbird-roberta-large/ --gradient_accumulation_steps 4

	"""
accuracy:  74.08%; (non-O)
accuracy:  90.51%; precision:  61.92%; recall:  71.49%; FB1:  66.36
            etime: precision:  66.67%; recall:  76.77%; FB1:  71.36  357
            fname: precision:  37.31%; recall:  49.18%; FB1:  42.43  402
        organizer: precision:  46.33%; recall:  56.58%; FB1:  50.95  436
      participant: precision:  68.17%; recall:  79.97%; FB1:  73.60  820
            place: precision:  66.75%; recall:  74.08%; FB1:  70.22  424
           target: precision:  42.07%; recall:  49.87%; FB1:  45.64  473
          trigger: precision:  75.23%; recall:  81.75%; FB1:  78.36  1292

# well, that good???

how about CRF
	"""


	# predict testset
# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" PYTHONPATH="." python task1/subtask1/transformer_models_sequence_classification.py --train_file ../20210312/subtask4-token/en-train.txt --test_file test_subtask4.txt --dest subtask-4-bigbird-large --model subtask-4-bigbird-base
