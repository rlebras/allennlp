
START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"


def valid_next_characters(function_calls, arg_numbers, last_token, valid_numbers, valid_variables,
                          valid_types):
    valid_variables.add('?')
    valid_variables.add('(')
    valid_numbers.add('?')
    valid_numbers.add('(')

    num_args = arg_numbers[-1]
    if last_token == END_SYMBOL:
        return {END_SYMBOL}
    elif last_token == '?':
        temp = valid_variables.copy()
        temp.remove('?')
        return temp
    elif last_token == START_SYMBOL:
        return {'('}
    elif last_token == ')' and len(function_calls) == 0:
        return {END_SYMBOL}
    elif last_token == '(':
        if len(function_calls) == 0:
            return {'Equals', 'And'}
        elif function_calls[-1] == 'And':
            return {'Equals'}
        elif function_calls[-1] == 'Equals':
            return {'Value', 'Rate', 'Times', 'Plus', 'Minus', 'Div'}
        elif function_calls[-1] == 'Join' or function_calls[-1] == 'Value' or \
                        function_calls[-1] == 'Rate':
            return {'Join'}
        elif function_calls[-1] == 'Minus' or function_calls[-1] == 'Plus' or function_calls[-1] == 'Times' or function_calls[-1] == 'Div':
            return {'Minus', 'Plus', 'Rate', 'Value', 'Times', 'Div'}
        else:
            raise Exception(str(function_calls) + '  aa   ' + str(arg_numbers))
    else:
        if function_calls[-1] == 'And' and 0 < num_args < 15:
            return {'(', ')'}
        elif function_calls[-1] == 'And' and num_args < 15:
            return {'('}
        elif function_calls[-1] == 'Equals' and num_args < 2:
            return valid_numbers.union(valid_variables)
        elif function_calls[-1] == 'Join' and num_args < 2:
            return valid_variables
        elif function_calls[-1] == 'Value' and num_args == 0:
            return valid_variables
        elif function_calls[-1] == 'Value' and num_args == 1:
            return valid_types
        elif function_calls[-1] == 'Rate' and num_args == 0:
            return valid_variables
        elif function_calls[-1] == 'Rate' and (num_args == 1 or num_args == 2):
            return valid_types
        elif (function_calls[-1] == 'Minus' or function_calls[-1] == 'Plus' or function_calls[-1] == 'Times' or function_calls[-1] == 'Div') and arg_numbers[-1] < 2:
            return valid_numbers.union(valid_variables)
        else:
            return {')'}


def update_state(function_calls, arg_numbers, last_token):
    operators = {'Equals', 'And', 'Value', 'Rate', 'Join', 'Plus', 'Minus', 'Times', 'Div'}
    if last_token == START_SYMBOL or last_token == END_SYMBOL:
        pass
    elif last_token == '?':
        pass
    elif last_token in operators:
        function_calls = function_calls + [last_token]
        arg_numbers = arg_numbers + [0]
    elif last_token == ')':
        function_calls = function_calls[:-1]
        arg_numbers = arg_numbers[:-1]
    else:
        arg_numbers = arg_numbers[:]
        arg_numbers[-1] += 1
    return function_calls, arg_numbers


if __name__ == '__main__':
    with open('../../../train.txt', 'r') as f:
        lines = f.read().splitlines()
        sentences = []
        for line in lines:
            sentences.append(line.split('\t')[1].split())

    for sentence in sentences:
        tokens = [START_SYMBOL] + sentence + [END_SYMBOL]
        function_calls = []
        num_args = [0]
        valid_variables = {'var' + str(i) for i in range(20)}
        valid_variables.add('(')
        valid_variables.add('?')
        valid_units = {'unit' + str(i) for i in range(20)}
        valid_numbers = {'num' + str(i) for i in range(20)}
        valid_numbers = {x for x in valid_numbers if x.startswith('num')}
        valid_numbers.add('(')
        valid_numbers.add('?')
        valid_actions = {START_SYMBOL}
        for i, token in enumerate(tokens):
            # if i < len(tokens) - 1 and tokens[i + 1] == END_SYMBOL:
                # print(valid_actions, token, function_calls, num_args, tokens)
            assert token in valid_actions, (valid_actions, token, function_calls, num_args, ' '.join(tokens))
            function_calls, num_args = update_state(function_calls, num_args, token)
            # print(function_calls, token)
            valid_actions = valid_next_characters(function_calls, num_args, token,
                                                  valid_numbers=valid_numbers,
                                                  valid_types=valid_units,
                                                  valid_variables=valid_variables)
        print('-' * 80)