import time
import json
from allennlp.prepare_seq2seq_data import is_num, nlp


with open('/Users/vidurj/allennlp/test.sh', 'r') as f:
    data = f.read().splitlines()

data = reversed(data)

results = []

count = 0

for x in data:
    if count > 0:
        results[-1].append(x)
        count -= 1
    elif x == 'wrong':
        count = 2
        results.append([])

# print(results)
question_num = {}
for x in results:
    a, b = list(reversed(x))
    # print(a, b[1:])
    question_num[a] = b[1:]

with open('/Users/vidurj/euclid/data/private/third_party/alg514/alg514_alignments.json',
          'r') as f:
    data = json.load(f)


for x in data:
    if x['iIndex'] in question_num:
        source = x['sQuestion'].replace('-', ' ')
        source_tokenized = [token.text for token in nlp(source)]
        numbers = [is_num(x) for x in source_tokenized if is_num(x) is not None]
        print('question number:', x['iIndex'])
        print('question:', x['sQuestion'])
        print('gold:', x['lSemantics'])
        pred = question_num[x['iIndex']]
        index = pred.split(',')[-1][:-1]
        top1 = ','.join(pred.split(',')[:-1]).split()
        final = []
        for x in top1:
            if x.startswith('num'):
                i = int(x[3:])
                final.append(str(numbers[i]))
            else:
                final.append(x)
        print('pred:', ' '.join(final))
        print('pred index:', index)
        print('-' * 100)
        print()