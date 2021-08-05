import json
from collections import Counter
import ast
import random
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_score

from allennlp.predictors.predictor import Predictor
from transformers import AlbertTokenizer, AlbertModel

CURRENT_DIRECTORY = Path(__file__).parent.resolve()
device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP:
    def __init__(self, data_path=None):
        if data_path is None:
            _coref_path = Path(CURRENT_DIRECTORY, '../..', 'data', 'subtask3-coreference', 'en-train.json').resolve()
            assert _coref_path.is_file()
            data_path = _coref_path
        self.data_path = data_path
        self.train, self.dev, self.test = None, None, None
        self.data_loaded = False
        self.model = Scorer()
        self.model = self.model.to(device)
        self.model.zero_grad()
        self.model_for_prepare = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)
        self.tokenizer_for_prepare = AlbertTokenizer.from_pretrained('albert-base-v2')

    def load_dataset(self):
        positives, negatives = [], []
        for p, n in get_instance_pairs(self.data_path):
            positives.append(p)
            negatives.append(n)
        positives, negatives = np.array(positives, dtype=object), np.array(negatives, dtype=object)
        indices = np.arange(len(positives))
        np.random.shuffle(indices)
        positives, negatives = positives[indices], negatives[indices]
        doc_no = len(positives)
        p_train, p_dev, p_test = positives[:int(doc_no * 0.70)], positives[
                                                                 int(doc_no * 0.70):int(doc_no * 0.95)], positives[
                                                                                                         int(doc_no * 0.95):]
        n_train, n_dev, n_test = negatives[:int(doc_no * 0.70)], negatives[
                                                                 int(doc_no * 0.70):int(doc_no * 0.95)], negatives[
                                                                                                         int(doc_no * 0.95):]

        self.train = prepare_set(p_train, n_train, model=self.model_for_prepare, tokenizer=self.tokenizer_for_prepare)
        self.dev = prepare_set(p_dev, n_dev, model=self.model_for_prepare, tokenizer=self.tokenizer_for_prepare)
        self.test = prepare_set(p_test, n_test, model=self.model_for_prepare, tokenizer=self.tokenizer_for_prepare)
        self.data_loaded = True

    def train(self):
        assert self.data_loaded is True
        epochs = 10
        bestloss = np.Inf
        bestprecision = 0
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005)

        for epoch in tqdm(range(1, epochs + 1), desc="Training MLP model"):
            print("Epoch", epoch)
            total_loss = 0
            random.shuffle(self.train)
            for i, (x, y, label) in enumerate(self.train):

                outputs = self.model(x, y)
                loss = criterion(outputs, label)
                total_loss += loss.item()
                loss.backward()

                if i % 32 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            print("Train loss:", total_loss)

            # calculate dev loss
            dev_loss = 0
            y_pred, y_dev = [], []
            for x, y, label in self.dev:
                outputs = self.model(x, y)
                y_pred.append(1 if outputs.item() >= 0.5 else 0)
                y_dev.append(label.item())
                loss = criterion(outputs, label)
                dev_loss += loss.item()

            print("Dev loss:", dev_loss)
            # if bestloss > dev_loss:
            #     bestloss = dev_loss
            #     torch.save(scorer_model.state_dict(), "scorer_model.pt")

            precision = precision_score(y_dev, y_pred)
            print("Dev precision:", precision)
            if precision > bestprecision:
                bestprecision = precision
                torch.save(self.model.state_dict(), "scorer_model.pt")

    def evaluate(self):
        scorer_model = self.model.load_state_dict(torch.load("scorer_model.pt"))
        scorer_model.eval()

        y_pred, y_train = [], []
        for x, y, label in self.train:
            y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0)
            y_train.append(label.item())
        print("Train :", classification_report(y_train, y_pred))
        print(confusion_matrix(y_train, y_pred))

        y_pred, y_dev = [], []
        for x, y, label in self.dev:
            y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0)
            y_dev.append(label.item())
        print("Dev :", classification_report(y_dev, y_pred))
        print(confusion_matrix(y_dev, y_pred))

        y_pred, y_test = [], []
        for x, y, label in self.test:
            y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0)
            y_test.append(label.item())
        print("Test :", classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        y_pred, y_true = [], []
        for x, y, label in (self.test + self.dev + self.train):
            y_pred.append(1)
            y_true.append(label.item())
        print("Baseline results :", classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))


def get_instance_pairs(file_name):
    """yields all possible pairs in a document"""
    with open(file_name, "r", encoding="utf-8") as fi:
        lines = fi.readlines()
        instances = [ast.literal_eval(line) for line in lines]

    for instance in instances:
        events = instance['event_clusters']
        sentence_ids = instance['sentence_no']
        sentences_all = instance['sentences']
        if len(events) < 1 or (len(events) == 1 and len(events[0]) < 2):
            continue

        positive_pairs = []
        for event_ids in events:
            sentence_id = [sentence_ids.index(event_id) for event_id in event_ids]
            sentences = [sentences_all[idx] for idx in sentence_id]
            for i in range(len(sentences) - 1):
                for j in range(i + 1, len(sentences)):
                    positive_pairs.append((sentences[i], sentences[j]))

        negative_pairs = []
        for i in range(len(events) - 1):
            for j in range(i + 1, len(events)):
                for s_1 in events[i]:
                    negative_pairs += [(sentences_all[sentence_ids.index(s_1)],
                                        sentences_all[sentence_ids.index(s_2)])
                                       for s_2 in events[j]]

        if len(positive_pairs + negative_pairs) == 0:
            continue

        yield positive_pairs, negative_pairs


def prepare_pair(x, y, label, tokenizer, max_length=256):
    """returns input_ids, input_masks, labels for pair of sentences in BERT input format"""
    x = tokenizer.encode_plus(x, pad_to_max_length=True, add_special_tokens=True,
                              max_length=max_length, truncation=True)
    y = tokenizer.encode_plus(y, pad_to_max_length=True, add_special_tokens=True,
                              max_length=max_length, truncation=True)
    x = (torch.tensor(x["input_ids"]).unsqueeze(0), torch.tensor(x["attention_mask"]).unsqueeze(0),
         torch.tensor(x["token_type_ids"]).unsqueeze(0))
    y = (torch.tensor(y["input_ids"]).unsqueeze(0), torch.tensor(y["attention_mask"]).unsqueeze(0),
         torch.tensor(y["token_type_ids"]).unsqueeze(0))
    label = torch.tensor(label).float()
    return x, y, label


def get_representation(x, model, hidden_layer=4):
    """returns sentence representation by pooling over hidden states of the model"""
    with torch.no_grad():
        x = tuple(i.to(device) for i in x)
        x_output = model(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2])
        averaged_hidden_states = torch.stack(x_output[2][-hidden_layer:]).mean(0)
        pooled = averaged_hidden_states[:, :x[1].sum(), :].mean(1)
    return pooled.clone().detach()


def prepare_set(p_set, n_set, model, tokenizer, max_length=256):
    pset = []
    for p, n in tqdm(zip(p_set, n_set), desc="Preparing dataset"):
        instance_pairs = []
        pairs = [prepare_pair(pair[0], pair[1], 1, max_length=max_length, tokenizer=tokenizer) for pair in p] + \
                [prepare_pair(pair[0], pair[1], 0, max_length=max_length, tokenizer=tokenizer) for pair in n]
        random.shuffle(pairs)
        for x, y, label in pairs:
            instance_pairs.append(
                (get_representation(x, model=model), get_representation(y, model=model), label.to(device)))

        # pset.append(instance_pairs)
        pset += instance_pairs
    return pset


class Scorer(torch.nn.Module):
    """MLP model used for pairwise scoring"""

    def __init__(self, input_size=1536):
        super(Scorer, self).__init__()
        self.lin = torch.nn.Linear(input_size, input_size // 2, bias=True)
        self.lin2 = torch.nn.Linear(input_size // 2, 1, bias=True)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.lin(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        return torch.sigmoid(x).squeeze()


class PretrainedSpanBERTModel:
    def __init__(self, data):
        model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        self.predictor = Predictor.from_path(model_url, cuda_device=1)
        self.data = data
        self.sentence_no_list = []
        self.event_clusters_list = []
        for d in data:
            self.sentence_no_list += d['sentence_no']
            self.event_clusters_list += d['event_clusters']

    def fit(self):
        pass

    def predict(self, data=None):
        if data is None:
            data = self.data
        preds = []

        r = list(range(len(data)))
        for idx in tqdm(r):
            instance = data[idx]
            id = instance["id"]
            pred_clusters, _ = self.predict_clusters(instance)
            if len(pred_clusters) != 0:
                pred = self.filter_predictions(pred_clusters, instance)
            else:
                pred = [[i] for i in instance["sentence_no"]]
            preds.append({"id": id, "pred_clusters": pred})
        return preds

    def filter_predictions(self, pred_clusters: List[list], instance) -> List[list]:
        all_elements_set = set(instance['sentence_no'])
        clusters_len = list(map(lambda x: len(set(x)), pred_clusters))
        longest_pred = set(pred_clusters[clusters_len.index(max(clusters_len))])
        remaining_elements_set = all_elements_set - longest_pred
        pred = [sorted(list(longest_pred))]
        if len(remaining_elements_set) != 0:
            standalone_no = self.standalone_clusters()
            processed = set([])
            for ele in remaining_elements_set:
                if ele not in processed:
                    if ele in standalone_no.keys():
                        ele_cluster_set = set(standalone_no[ele])
                        ele_cluster = ele_cluster_set and remaining_elements_set
                        processed = ele_cluster or processed
                        pred.append(sorted(list(ele_cluster)))
            remaining_elements_set = remaining_elements_set - processed
            if len(remaining_elements_set) != 0:
                pred.append(sorted(list(remaining_elements_set)))
        return pred

    def predict_clusters(self, data):
        sentences = data['sentences']

        _pred = self.predictor.predict(document='\n'.join(sentences))
        _clusters = _pred['clusters']
        _docu = _pred['document']
        sentences_no_mark = np.zeros(len(_docu), dtype=int)
        _split_idx = [i for i, e in enumerate(_docu) if e == '\n'] + [len(_docu) - 1]
        begin_idx = 0
        for i, end_idx in enumerate(_split_idx):
            sent_no = data['sentence_no'][i]
            sentences_no_mark[begin_idx: end_idx + 1] = sent_no
            begin_idx = end_idx
        sentences_no_mark = sentences_no_mark.tolist()
        _sent_clusters = []
        _sent_clusters_words = []
        for cs in _clusters:
            _sent_cluster = []
            _sent_cluster_words = []
            for c in cs:
                try:
                    _sent_cluster.append(sentences_no_mark[c[0]])
                except IndexError:
                    raise
                if c[0] == c[1]:
                    _sent_cluster_words.append(_docu[c[0]])
                else:
                    _sent_cluster_words.append(' '.join(_docu[c[0]:c[1] + 1]))
            # _sent_cluster = sorted(list(set(_sent_cluster)))
            _sent_clusters.append(_sent_cluster)
            _sent_clusters_words.append(_sent_cluster_words)
        return _sent_clusters, _sent_clusters_words

    def standalone_clusters(self):
        cooccurrence_df = self.cooccurrence_matrix()
        occur_alone = cooccurrence_df[cooccurrence_df == 0].dropna().index.to_list()
        occur_alone = {sent_no: [sent_no] for sent_no in occur_alone}
        sentence_no_counter = dict(Counter(self.sentence_no_list))
        occur_once = dict(filter(lambda elem: elem[1] == 1, sentence_no_counter.items())).keys()
        for cluster in self.event_clusters_list:
            for s_no in cluster:
                if s_no in occur_once:
                    occur_alone[s_no] = cluster
        return occur_alone

    def cooccurrence_matrix(self, fill_diag=False):
        sentence_no_range = range(1, max(self.sentence_no_list) + 1)
        results = pd.DataFrame(index=sentence_no_range, columns=sentence_no_range)
        results = results.fillna(0)
        sentence_no_counter = dict(Counter(self.sentence_no_list))

        # Find the co-occurrences:
        for i in self.event_clusters_list:
            for j in range(len(i)):
                for item in i[:j] + i[j + 1:]:
                    # occurrences[l[i]][item] += 1
                    _tmp = results.at[i[j], item] + 1
                    results.at[i[j], item] = _tmp
        if fill_diag:
            for i in sentence_no_range:
                _idx = i - 1
                results[_idx][_idx] = sentence_no_counter[i]
        return results


class MaxClusterModel:
    def __init__(self):
        pass

    def fit(self, data):
        """
        Doesn't use the given data.
        Returns nothing.
        """
        return

    def predict(self, data):
        """
        Takes some data (.json) and makes predictions.
        Simply puts all sentences in a single cluster.
        """
        preds = []
        for idx, instance in enumerate(data):
            preds.append({"id": instance["id"], "pred_clusters": [instance['sentence_no']]})

        return preds


class MinClusterModel:
    def __init__(self):
        pass

    def fit(self, data):
        """
        Doesn't use the given data.
        Returns nothing.
        """
        return

    def predict(self, data):
        """
        Takes some data (.json) and makes predictions.
        Simply puts all sentences in a single cluster.
        """
        preds = []
        for idx, instance in enumerate(data):
            preds.append({"id": instance["id"], "pred_clusters": [[instance['sentence_no']]]})

        return preds


if __name__ == '__main__':
    test_model = MLP()
    test_model.load_dataset()
    test_model.train()
    test_model.evaluate()
