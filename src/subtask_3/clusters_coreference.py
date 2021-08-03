import argparse
import itertools
import json
import subprocess
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from allennlp.predictors.predictor import Predictor
from icecream import ic
from tqdm import tqdm


class SpanBERTModel:
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
            ic(id, pred_clusters)
            if len(pred_clusters) != 0:
                pred = self.filter_predictions(pred_clusters, instance)
            else:
                pred = [[i] for i in instance["sentence_no"]]
            ic(pred)
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


class OneClusterModel:
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


def convert_to_scorch_format(docs, cluster_key="event_clusters"):
    # Merge all documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(docs):
        for cluster in doc[cluster_key]:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_events = [event for cluster in all_clusters for event in cluster]
    all_links = sum([list(itertools.combinations(cluster, 2)) for cluster in all_clusters], [])

    return all_links, all_events


def evaluate(goldfile, sysfile):
    """
    Uses scorch -a python implementaion of CoNLL-2012 average score- for evaluation. > https://github.com/LoicGrobol/scorch | pip install scorch
    Takes gold file path (.json), predicted file path (.json) and prints out the results.
    This function is the exact way the subtask3's submissions will be evaluated.
    """
    gold = read(goldfile)
    sys = read(sysfile)

    gold_links, gold_events = convert_to_scorch_format(gold)
    sys_links, sys_events = convert_to_scorch_format(sys, cluster_key="pred_clusters")

    with open("gold.json", "w") as f:
        json.dump({"type": "graph", "mentions": gold_events, "links": gold_links}, f)
    with open("sys.json", "w") as f:
        json.dump({"type": "graph", "mentions": sys_events, "links": sys_links}, f)

    subprocess.run(["scorch", "gold.json", "sys.json", "results.txt"])
    print(open("results.txt", "r").read())
    subprocess.run(["rm", "gold.json", "sys.json", "results.txt"])


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="The path to the training data json file")
    parser.add_argument('-prediction_output_file', '--prediction_output_file', required=True,
                        help="The path to the prediction output json file")
    parser.add_argument('-test_file', '--test_file', required=False, help="The path to the test data json file")
    args = parser.parse_args()
    return args


def main(train_file, prediction_output_file, test_file=None, submission_file=None):
    # Read training data.
    train_data = read(train_file)

    # Create model.
    model = SpanBERTModel(train_data)

    # Fit.
    model.fit()

    # Predict train data so that we can evaluate our system
    train_predictions = model.predict()
    with open(prediction_output_file, "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")

    # Evaluate sys outputs and print results.
    evaluate(train_file, prediction_output_file)

    # If there is a test file provided, make predictions and write them to file.
    if test_file:
        # Read test data.
        test_data = read(test_file)
        # Predict and save your predictions in the required format.
        test_predictions = model.predict(test_data)
        if not submission_file:
            submission_file = "sample_submission.json"
        with open(submission_file, "w", encoding="utf-8") as f:
            for doc in test_predictions:
                f.write(json.dumps(doc) + "\n")


ic.disable()
# args = parse()
# main(args.train_file, args.prediction_output_file, args.test_file)
train_file = "../data/subtask3-coreference/en-train.json"
prediction_output_file = "outputs/subtask3-predictions-OneCluster.json"
test_file = "../data/test/test_subtask3.json"
submission_file = "outputs/subtask3-submission-OneCluster.json"
# main(train_file, prediction_output_file, test_file, submission_file)
train_data = read(train_file)
model = OneClusterModel()
model.fit(train_data)
# Predict train data so that we can evaluate our system
train_predictions = model.predict(train_data)
with open(prediction_output_file, "w", encoding="utf-8") as f:
    for doc in train_predictions:
        f.write(json.dumps(doc) + "\n")
# Evaluate sys outputs and print results.
evaluate(train_file, prediction_output_file)

test_data = read(test_file)
# Predict and save your predictions in the required format.
test_predictions = model.predict(test_data)
if not submission_file:
    submission_file = "sample_submission.json"
with open(submission_file, "w", encoding="utf-8") as f:
    for doc in test_predictions:
        f.write(json.dumps(doc) + "\n")
