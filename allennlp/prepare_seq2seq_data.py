import json
import random
import traceback
from collections import Counter, defaultdict
import spacy
import itertools

from allennlp.type_checking import valid_next_characters, update_state, \
    START_SYMBOL, END_SYMBOL

nlp = spacy.load('en')
PRECISION = 7

meaningful_units = {
    # 'Dollar',
    # 'Pound',
    # 'Pound',
    # 'Hour',
    # 'Mile',
    # 'Point',
    # 'Coin',
    # 'Gallon',
    # 'Liter',
    # 'Milliliter',
    # 'Minute',
    # 'Ounce',
    # 'Foot',
    # 'Kilogram',
    # 'Ton',
    # 'Inch',
    # 'Cent',
    # 'Animal'
}

NUMBER_WORDS = {
    'twice': 2,
    'thrice': 3,
    'double': 2,
    'triple': 3,
    'quadruple': 4,
    'quituple': 5,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000
}


def is_strict_num(string):
    try:
        float(string.replace(',', ''))
        return True
    except:
        return False


def is_num(string):
    string = string.lower()
    if string in NUMBER_WORDS:
        return NUMBER_WORDS[string]
    try:
        temp = float(string.replace(',', ''))
        temp = round(temp, PRECISION)
        return temp
    except:
        return None


def standardize_question(text):
    source = text.replace('-', ' ')
    source_tokenized = [token.text for token in nlp(source)]
    number_to_tokens = defaultdict(list)
    for index, token in enumerate(source_tokenized):
        number = is_num(token)
        if number is not None:
            number_to_tokens[number].append('num' + str(index))
    return ' '.join(source_tokenized), number_to_tokens


def standardize_logical_form_with_validation(text, number_to_tokens, randomize):
    remaining_variable_names = ['var' + str(i) for i in range(10)]
    remaining_units = ['unit' + str(i) for i in range(2)]
    if randomize:
        random.shuffle(remaining_variable_names)
        random.shuffle(remaining_units)
    target_tokens = text.replace('?', ' ? ').replace('(', ' ( ').replace(')', ' ) ').split()
    var_assignments = {}
    type_assignments = {}
    standardized_tokens = []
    num_open_parens = 0
    num_close_parens = 0
    function_calls = []
    arg_numbers = [0]
    last_token = START_SYMBOL
    for token in target_tokens:
        valid_tokens = valid_next_characters(function_calls,
                                             arg_numbers,
                                             last_token=last_token,
                                             valid_numbers={'NUMBER'},
                                             valid_variables={'VARIABLE'},
                                             valid_types={'TYPE'})
        assert num_close_parens <= num_open_parens, (num_open_parens, num_close_parens)
        if token == '(':
            num_open_parens += 1
        elif token == ')':
            num_close_parens += 1
        elif is_strict_num(token):
            assert 'NUMBER' in valid_tokens
            number = is_num(token)
            if number in number_to_tokens:
                tokens = number_to_tokens[number]
                token = random.choice(tokens)
            else:
                raise RuntimeError(
                    'Number {} not in number_to_token {}.'.format(number, number_to_tokens))
        elif token == '?':
            pass
        elif 'TYPE' in valid_tokens:
            assert 'VARIABLE' not in valid_tokens and 'NUMBER' not in valid_tokens
            if token not in type_assignments:
                assert token not in var_assignments, '{} token in var assignments too!'.format(
                    token)
                # TODO preserving types
                if token in meaningful_units:
                    type_assignments[token] = 'unit' + token
                else:
                    type_assignments[token] = remaining_units.pop(0)
            token = type_assignments[token]
        elif 'VARIABLE' in valid_tokens:
            if token not in var_assignments:
                assert token not in type_assignments, '{} token in type assignments too!'.format(
                    token)
                var_assignments[token] = remaining_variable_names.pop(0)
            token = var_assignments[token]

        assert token.startswith('var') or token.startswith('unit') or token.startswith(
            'num') or token in valid_tokens, (token, valid_tokens, text)
        standardized_tokens.append(token)
        function_calls, arg_numbers = update_state(function_calls, arg_numbers, token)
        last_token = token
    valid_tokens = valid_next_characters(function_calls,
                                         arg_numbers,
                                         last_token=last_token,
                                         valid_numbers={'NUMBER'},
                                         valid_variables={'VARIABLE'},
                                         valid_types={'TYPE'})
    assert valid_tokens == {END_SYMBOL}, (valid_tokens, standardized_tokens, text)
    return ' '.join(standardized_tokens), (var_assignments, type_assignments)


def prepare_synthetic_data():
    number_tokens = [str(i * 100) for i in range(1, 100)]

    def sample_question(size, objects):
        question = ['Tom', 'has']
        random.shuffle(number_tokens)
        question_index = random.randint(0, size - 1)
        for index in range(size):
            if index == question_index:
                solution = len(question)
            question.append(number_tokens[index])
            question.append(objects[index])
            question.append(',')
        question = question[:-1]
        question.extend(['.', 'How', 'many', objects[question_index], 'does', 'he', 'have', '?'])
        assert question[solution] == number_tokens[question_index]
        return ' '.join(question), 'num' + str(solution)

    with open('objects.txt', 'r') as f:
        tokens = f.read().lower().split()
    objects = []
    for token in tokens:
        if len(token) > 2:
            objects.append(token)
    print(len(objects))
    train_objects = objects[:-100]
    train = []
    questions = []
    answers = []
    # for _ in range(10000):
    #     size = random.randint(2, 5)
    #     random.shuffle(train_objects)
    #     question, answer = sample_question(size, train_objects)
    #     train.append(question + '\t' + answer)
    #     questions.append(question)
    #     answers.append(answer)
    #
    # with open('synthetic_train.txt', 'w') as f:
    #     f.write('\n'.join(train))
    #
    # with open('synthetic_train.json', 'w') as f:
    #     f.write('\n'.join(questions))
    #
    # with open('synthetic_train_solutions.txt', 'w') as f:
    #     f.write('\n'.join(answers))

    dev = []
    dev_objects = objects[-100:]
    answers = []
    questions = []
    for _ in range(1000):
        size = random.randint(6, 20)
        random.shuffle(dev_objects)
        question, answer = sample_question(size, dev_objects)
        dev.append(question + '\t' + answer)
        questions.append(json.dumps({'source': question}))
        answers.append(answer)

    with open('synthetic_dev.txt', 'w') as f:
        f.write('\n'.join(dev))

    with open('synthetic_dev.json', 'w') as f:
        f.write('\n'.join(questions))

    with open('synthetic_dev_solutions.txt', 'w') as f:
        f.write('\n'.join(answers))


def write_data(data, file_name, num_iters, randomize, silent=True):
    lines = []
    number_to_tokens = []
    question_numbers = []
    original_units = []
    for _ in range(num_iters):
        for question_number, question in enumerate(data):
            if question['lSemantics'] == '':
                continue
            # if question['iIndex'] != '6226':
            #     continue
            source, number_to_token = standardize_question(question['sQuestion'])
            try:
                target, (_, type_assignments) = standardize_logical_form_with_validation(
                    question['lSemantics'],
                    number_to_token,
                    randomize=randomize)
                original_units.extend(type_assignments.keys())
                lines.append(source + '\t' + target)
                number_to_tokens.append(number_to_token)
            except:
                if not silent:
                    print(question)
                    traceback.print_exc()
                continue

    assert len(number_to_tokens) == len(lines), (len(number_to_tokens), len(lines))
    # counts = list(Counter(original_units).items())
    # counts.sort(key=lambda x: - x[1])
    # for k, v in counts:
    #     print(k, v)
    print('num data points', len(lines))
    print('-' * 70)
    with open(file_name, 'w') as f:
        f.write('\n'.join(lines))
        strs = []
    for line in lines:
        strs.append(json.dumps({'source': line.split('\t')[0]}))

    with open(file_name[:-4] + '.json', 'w') as f:
        f.write('\n'.join(strs))

    with open(file_name[:-4] + '.question_numbers', 'w') as f:
        f.write('\n'.join(question_numbers))


def json_from_text_file(input_path='../euclid/data/private/rate_facts.txt',
                        output_path='allennlp/additional_annotations.json'):
    points = []
    with open(input_path, 'r') as f:
        data = f.read().strip().splitlines()
    for line in data:
        print(line)
        question, logical_form = line.split('\t')
        points.append({'sQuestion': question.strip(), 'lSemantics': logical_form.strip()})

    with open(output_path, 'w') as f:
        json.dump(points, f)


def create_sentence_aligned_data(alignments):
    import itertools
    with open('/Users/vidurj/euclid/data/private/third_party/alg514/kushman_annotated.json',
              'r') as f:
        data = json.load(f)

    key_to_sentence = {}
    for q in data:
        for sentence_number, sentence in enumerate(q['nlp']['sentences']):
            key = (q['iIndex'], sentence_number)
            assert key not in key_to_sentence
            key_to_sentence[key] = sentence['text']['content']

    question_to_sentence_semantics = defaultdict(lambda: [[] for _ in range(10)])
    for q in alignments:
        if 'lAlignments' in q:
            lines = q['lAlignments']
            for substring in lines:
                sentence_number, semantics = substring.split(' -> ')
                sentence_number = int(sentence_number)
                if sentence_number >= 0:
                    question_to_sentence_semantics[q['iIndex']][sentence_number].append(semantics)

    problems = []
    print(key_to_sentence.keys())
    for q_index, semantics in question_to_sentence_semantics.items():
        valid = [(i, l) for i, l in enumerate(semantics) if len(l) > 0]
        for size in range(1, len(valid)):
            for sequence in itertools.combinations(valid, size):
                sequence = list(sequence)
                sequence.sort(key=lambda x: x[0])
                final_sentence = []
                final_logical_form = []
                for (i, l) in sequence:
                    final_sentence.append(key_to_sentence[(int(q_index), i)])
                    final_logical_form.extend(l)

                num_forms = len(final_logical_form)
                final_logical_form = ' '.join(final_logical_form)
                final_sentence = ' '.join(final_sentence)
                if num_forms > 1:
                    final_logical_form = '( And ' + final_logical_form + ' )'
                problem = {
                    'iIndex': q_index,
                    'sQuestion': final_sentence,
                    'lSemantics': final_logical_form
                }
                problems.append(problem)
    return problems


if __name__ == '__main__':
    # prepare_synthetic_data()
    with open('/Users/vidurj/euclid/data/private/third_party/alg514/alg514_alignments.json',
              'r') as f:
        data = json.load(f)

    with open('allennlp/additional_annotations.json', 'r') as f:
        additional_data = json.load(f)

    all_train_subsets = create_sentence_aligned_data(data[:-100])
    write_data(all_train_subsets + data[:-100], 'train.txt', randomize=True, num_iters=3)
    write_data(data[-100:], 'dev.txt', randomize=False, num_iters=1)
    write_data(data[-100:], 'test.txt', randomize=True, num_iters=1)
