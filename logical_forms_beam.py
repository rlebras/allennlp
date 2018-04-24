import json
import sys

with open(sys.argv[2], 'r') as f:
    lines = f.read().splitlines()

question_tokens = [line.split('\t')[0].split() for line in lines]

with open(sys.argv[1], 'r') as f:
    lines = f.read().splitlines()
assert len(lines) == len(question_tokens), (len(lines), len(question_tokens))
preds = []
for _line, tokens in zip(lines, question_tokens):
    programs = _line.replace('"', '').replace("\\n", '\n').replace('\n***\n', '').split('\n')
    for program in programs:
        processed_tokens = []
        for token in json.loads(program)['predicted_tokens']:
            if token.isdigit():
                token = tokens[int(token)]
            processed_tokens.append(token)
        preds.append(' '.join(processed_tokens))

with open(sys.argv[3], 'w') as f:
    f.write('\n'.join(preds))
