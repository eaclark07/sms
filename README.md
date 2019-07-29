# Sentence Mover's Similarity

This is the code and data for the [sentence mover's similarity metrics](https://www.aclweb.org/anthology/P19-1264).

The code is based on the Word Mover's Distance implementation from [this repo](https://github.com/src-d/wmd-relax) and [this paper](http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf). Make sure you follow installation info for this repo before running SMS code.

## Running Instructions

Input should be a file of tab-separated reference and hypothesis texts, one pair per line.

The file should be passed in, along with the word embedding type (`glove` or `elmo`) and the metric type (`wms`, `sms`, or `s+wms`).

Output will be written to the input's directory, labeled with the embedding and metric choices.

e.g., `python smd.py input.tsv glove sms` will calculate the SMS numbers for the file *input.tsv* using GloVe embeddings. The output will be written to *input_glove_sms.out*.
