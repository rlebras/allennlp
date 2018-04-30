import scipy
import atexit
import os
import signal
import subprocess
import time
import sys
import cmd
import sys
import torch
import random

# class Interpreter(cmd.Cmd):
#     intro = 'Welcome. Type help or ? to list commands.\n'
#     prompt = '(teacher)'
#     file = None
#
#     def do_solve(self, arg):
#         print('solving', arg)
#
#     def do_learn(self, arg):
#         print('learning', arg)
#
#     def do_print_foo(self, _):
#         print('foo')
#         while sys.stdin.read() == '':
#             time.sleep(0.1)
#
#     def complete_solve(self, text, line, begidx, endidx):
#         print('-' * 10)
#         print('text:', text)
#         print('line:', line)
#         print('begidx:', begidx)
#         print('endidx:', endidx)
#         print('-' * 10)
#         return []
#
# Interpreter().cmdloop()
#
# print(os.environ)
# print('agent_name' in os.environ)
# print('AGENT_NAME' in os.environ)



# A = torch.FloatTensor([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5]]])
# ind = torch.FloatTensor([0, 1, 0])
#
#
# # def batched_index_select(t, dim, inds):
# #     dummy = inds.unsqueeze(1).unsqueeze(2).expand(inds.size(0), 1, t.size(2))
# #     print(dummy.size())
# #     out = t.gather(dim, dummy)  # b x e x f
# #     return out.squeeze(1)
# #
# # dim = 1
# # print(batched_index_select(A, dim, ind))
#
# tensor_1 = torch.FloatTensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
# tensor_2 = torch.FloatTensor([0, 1, 0])
# print((A * ind).sum(dim=-1))


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def f(x, y):
    return abs(x - y) + abs(x) + abs(y)

def grad(x, y):
    if x > y:
        return (2, 0)
    elif y > x:
        return (0, 2)
    else:
        return (0, 0)


def f2(x, y, a):
    return abs(a * x - a * y) + abs(a * x) + abs(a * y)

def grad2(x, y, a):
    if x > y:
        return (2, 0, sign(a) * f(x, y))
    elif y > x:
        return (0, 2, sign(a) * f(x, y))
    else:
        return (0, 0, sign(a) * f(x, y))


def optimize(f, g, lr=2):
    cur = [100, 101]
    for _ in range(100):
        assert all([x >= 0 for x in cur[:2]]), cur
        print(f(*cur))
        grad = g(*cur)
        cur = [x - lr * y for x, y in zip(cur, grad)]


# optimize(f, grad)
a = ['The', 'sum', 'of', '0.75', 'of', 'a', 'number', 'and', 'two', 'is', 'eight', '.', 'Find', 'the', 'number', '.']
print(' '.join(a))



