"""


Author: Tong
Time: 14-03-2020
"""

import tensorflow as tf
import json


def read_examples(*dataset_paths):
    examples = []
    dataset = tf.data.TFRecordDataset(dataset_paths)
    for raw_sentence in dataset:
        sentence = tf.train.Example()
        sentence.ParseFromString(raw_sentence.numpy())
        examples.append(sentence)
    return examples


def get_feature(sentence, feature_name, idx=0):
    feature = sentence.features.feature[feature_name]
    # print(sentence.features.feature)
    return getattr(feature, feature.WhichOneof('kind')).value[idx]


"sentence"
"relation_name"
"relation_id"

"num_pos_raters"
"num_raters"

"url"
"source_name"
"target_name"


for i in ["5", "21"]:
    file_in = "data/webred_{i}.tfrecord".format(i=i)
    file_out = "data/webred_{i}.json".format(i=i)
    webred_sentences = read_examples(file_in)
    data = []
    for sentence in webred_sentences:
        item = {"sentence": get_feature(sentence, "sentence").decode('utf-8'),
                "relation_name": get_feature(sentence, "relation_name").decode('utf-8'),
                "relation_id": get_feature(sentence, "relation_id").decode('utf-8'),
                "url": get_feature(sentence, "url").decode('utf-8'),
                "source_name": get_feature(sentence, "source_name").decode('utf-8'),
                "target_name": get_feature(sentence, "target_name").decode('utf-8'),
                "num_pos_raters": get_feature(sentence, 'num_pos_raters'),
                "num_raters": get_feature(sentence, 'num_raters')}
        data.append(item)
    with open(file_out, "w") as outf:
        json.dump(file_out, outf)
