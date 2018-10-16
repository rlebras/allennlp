import json
import csv

dataFolder = "/data/"

def getFields(fname, list_fields, delim = ','):
    with open(fname) as f:
        reader = csv.DictReader(f, delimiter=delim)
        #        data = [r for r in reader]

        # Convert all the fields to list of strings, except for event
        data = []
        for r in reader:
            dict = {}
            for key, value in r.items():
                if key in list_fields:
                    if r[key] == 'NONE':
                        dict[key] = []
                    else:
                        dict[key] = eval(r[key])
                if key == 'event':
                    dict[key] = r[key]
            data.append(dict)
    return data

def temp():
    fieldsWN = ['xNeed', 'xWant', 'oWant']
    dataWN = getFields("../data/wantneed.csv", fieldsWN, ';')
    fieldsE = ['xEffect', 'oEffect']
    dataE = getFields("../data/effects.csv", fieldsE)
    fieldsA = ['xAttr']
    dataA = getFields("../data/attr.csv", fieldsA)
    data = dataWN + dataE + dataA
    print(data)
    fields = ['event'] + fieldsWN + fieldsE + fieldsA

    sorted_data = sorted(data, key=lambda x: x['event'])

    split_1 = int(0.8 * len(sorted_data))
    split_2 = int(0.9 * len(sorted_data))
    train = sorted_data[:split_1]
    dev = sorted_data[split_1:split_2]
    test = sorted_data[split_2:]
    writecsv('../data/train.txt', train, fields)
    writecsv('../data/dev.txt', dev, fields)
    writecsv('../data/test.txt', test, fields)

import xmltodict
def loadCOPA():
    filename = dataFolder + 'copa/copa-test.xml'

    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())

    items = doc['copa-corpus']['item']
    #for s in items:
        #print(s['p'])
        #print(s['@most-plausible-alternative'])
        #print(s['@asks-for'])
        #print(s['a1'])
        #print(s['a2'])
    return items

def loadE2E():
    filename = dataFolder + 'event2event_data/v2/all.csv'
    ret = []
    with open(filename, "r") as data_file:
        reader = csv.DictReader(data_file)
        header = reader.fieldnames

        target_fields = ["xEffect", "oEffect", "xWant", "oWant"]
        for (line_num, line_dict) in enumerate(reader):
            # source_seq : event
            event = line_dict["event"]
            xWant = line_dict["xWant"]
            oWant = line_dict["oWant"]
            xEffect = line_dict["xEffect"]
            oWant = line_dict["oWant"]

            retval = {'event': event}
            target_indices = range(0, len(target_fields))
            targets = []
            for field in target_fields:
                if line_dict[field] == None or line_dict[field] == "[]":
                    retval[field] = []
                else:
                    retval[field] = json.loads(line_dict[field])
            ret.append(retval)
    return ret

import gensim
print(gensim.__version__)

def loadW2V():

    # Load Google's pre-trained Word2Vec model.
    return gensim.models.KeyedVectors.load_word2vec_format(dataFolder + 'word2vec/GoogleNews-vectors-negative300.bin', binary=True)

import numpy as np
from scipy import spatial


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def sentence_similarity(s1, s2, model, index2word_set):
    s1_afv = avg_feature_vector(s1, model=model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(s2, model=model, num_features=300, index2word_set=index2word_set)
    if np.sum(np.square(s1_afv)) == 0:
        return 0
    if np.sum(np.square(s2_afv)) == 0:
        return 0
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    return sim


def compareCOPAwE2E(copa_entry, alt, e2e_entry, target, field, model, index2word_set):
    if target == "":
        return 0

    if copa_entry['@asks-for'] == 'cause':
        copa_a = copa_entry[alt]
        copa_b = copa_entry['p']
    else:
        copa_a = copa_entry['p']
        copa_b = copa_entry[alt]

    if field == 'xEffect' or field == 'oEffect':
        e2e_a = e2e_entry['event']
        e2e_b = target
    else:
        e2e_a = target
        e2e_b = e2e_entry['event']

    s_a = sentence_similarity(copa_a, e2e_a, model, index2word_set)
    s_b = sentence_similarity(copa_b, e2e_b, model, index2word_set)
    return s_a + s_b


def NearestCausalRelation():
    print("Let's do it!!")
    copa = loadCOPA()
    print("Loading W2V")
    model = loadW2V()
    print("Done loading W2V")

    index2word_set = set(model.wv.index2word)

    #sim = sentence_similarity('this is a sentence', 'this is not a sentense', model, index2word_set)
    #print(sim)

    #sim = sentence_similarity('this is a sentence', 'A dog walking', model, index2word_set)
    #print(sim)

    e2e = loadE2E()

    nb_correct = 0
    nb_total = 0
    for i, item in enumerate(copa):
        print("p:  ", item['p'])
        print("a1: ", item['a1'])
        print("a2: ", item['a2'])
        print("sol:", item['@most-plausible-alternative'])
        best_a1 = ''
        best_a1_score = 0
        best_a2 = ''
        best_a2_score = 0
        for e in e2e:
            for effect in e['xEffect']:
                a1_score = compareCOPAwE2E(item, 'a1', e, effect, 'xEffect', model, index2word_set)
                if a1_score > best_a1_score:
                    best_a1_score = a1_score
                    best_a1 = e
                a2_score = compareCOPAwE2E(item, 'a2', e, effect, 'xEffect', model, index2word_set)
                if a2_score > best_a2_score:
                    best_a2_score = a2_score
                    best_a2 = e

            for cause in e['xWant']:
                a1_score = compareCOPAwE2E(item, 'a1', e, cause, 'xWant', model, index2word_set)
                if a1_score > best_a1_score:
                    best_a1_score = a1_score
                    best_a1 = e
                a2_score = compareCOPAwE2E(item, 'a2', e, cause, 'xWant', model, index2word_set)
                if a2_score > best_a2_score:
                    best_a2_score = a2_score
                    best_a2 = e

        print("best_a1_score: ", best_a1_score)
        print("best_a1:       ", best_a1)
        print("best_a2_score: ", best_a2_score)
        print("best_a2:       ", best_a2)

        nb_total = nb_total + 1
        if item['@most-plausible-alternative'] == '1':
            if best_a1_score >= best_a2_score:
                nb_correct = nb_correct + 1
                print("+++ Correct")
            else:
                print("--- Incorrect")
        else:
            if best_a2_score >= best_a1_score:
                nb_correct = nb_correct + 1
                print("+++ Correct")
            else:
                print("--- Incorrect")

        print("Nb correct: ", nb_correct)
        print("Nb total: ", nb_total)
        print("Accuracy: ", nb_correct*1.0/nb_total)

    print("[Final] Nb correct: ", nb_correct)
    print("[Final] Nb total: ", nb_total)
    print("[Final] Accuracy: ", nb_correct*1.0/nb_total)


def main():
    NearestCausalRelation()

import json
def writejson(data):
    #with open('snli_1.0_test_for_pred.txt', 'w') as csvfile:
    #    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    #    for wp in data:
    #        csvwriter.writerow(wp)

    with open('testing_output.txt', 'w') as fp:
        for p in data:
            json.dump(p, fp)
            fp.write('\n')
import csv
def writecsv(filename, data, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            formatted_d = {}
            for key, val in d.items():
                formatted_d[key] = json.dumps(val)
            writer.writerow(formatted_d)

if __name__ == "__main__":
    main()
