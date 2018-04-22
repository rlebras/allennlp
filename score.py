import json
import sys


with open(sys.argv[1], 'r') as f:
    preds = f.read().splitlines()

preds = ['num' + json.loads(x)['predicted_tokens'][0] for x in preds]


with open(sys.argv[2], 'r') as f:
    gold = f.read().split()

assert len(gold) == len(preds), (len(gold), len(preds))

print(preds[:20])
print(gold[:20])

print(sum([x == y for x, y in zip(preds, gold)]) / len(gold))