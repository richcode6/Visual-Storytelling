from nltk import *
import json
import collections

vocab = {"<pad>": 0,
 "<bos>": 1,
 "<eos>": 2,
 "<unk>": 3}

vocabCounter = collections.Counter()

files = {"extractedDescriptions_train.txt", "extractedDescriptions_val.txt", "extractedStory_train.txt", "extractedStory_val.txt"}

index = 4



for f in files:
    fileOpen = open(f, "r")
    lines = fileOpen.readlines()
    for line in lines:
        sents = sent_tokenize(line.strip())
        for sentence in sents:
            words = word_tokenize(sentence)
            for word in words:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                    vocabCounter[word] = 1
                vocabCounter[word] += 1

print("Vocab Len:: ", len(vocab))

#commonWords = vocabCounter.most_common()

for entry in vocabCounter:
    if vocabCounter[entry] < 10:
        del vocab[entry]

print("Vocab Len -- updated:: ", len(vocab))

with open("Vs_vocab_clean.txt", "w") as file:
    file.write(json.dumps(vocab))

#with open("Vs_vocab_counter.txt", "w") as file:
#    file.write(json.dumps(vocabCounter))