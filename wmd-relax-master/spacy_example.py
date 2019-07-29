# sys.argv[1:] defines Wikipedia page titles
# This example measures WMDs from the first page to all the rest

from collections import Counter
import sys

import numpy
import spacy
import requests
from wmd import WMD

from spacy.lang.en.stop_words import STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
print("loading spaCy")
# nlp = spacy.load("en")
nlp = spacy.load('en_core_web_md')

stops = STOP_WORDS


# List of page names we will fetch from Wikipedia and query for similarity
titles = sys.argv[1:] or ["Germany", "Spain", "Google"]

documents = {}
for title in titles:
    print("fetching", title)

    pages = requests.get(
        "https://en.wikipedia.org/w/api.php?action=query&format=json&titles=%s"
        "&prop=extracts&explaintext" % title).json()["query"]["pages"]
    print("parsing", title)

    doctext = next(iter(pages.values()))["extract"]
    doctext = ' '.join(doctext.split(' ')[:30])

    text = nlp(doctext)
    tokens = [t for t in text if t.is_alpha and not t.is_stop]

    words = Counter(t.text for t in tokens)
    orths = {t.text: t.orth for t in tokens}
    sorted_words = sorted(words)
    documents[title] = (title, [orths[t] for t in sorted_words],
                        numpy.array([words[t] for t in sorted_words],
                                    dtype=numpy.float32))

# Hook in WMD
class SpacyEmbeddings(object):
    def __getitem__(self, item):
        return nlp.vocab[item].vector

embeddings = SpacyEmbeddings()

vocabulary_min = 10
calc = WMD(embeddings, documents, vocabulary_min=vocabulary_min)

print("calculating")
# Germany shall be closer to Spain than to Google

neigbors_of_germany = calc.nearest_neighbors(titles[0])

for title, relevance in neigbors_of_germany:
    print("%24s\t%s" % (title, relevance))


