from transformers import BertTokenizer, BertModel
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random
import json

# a = np.arange(150).repeat(100)
# b = np.arange(150).repeat(100)
#
# split = []
# chunk_size = 3
# for i in np.unique(a):
#     print(i)
#     print(np.where(b==i)[0])
#     c = np.where(b==i)
#
#     d = np.reshape(c[0][0: int(c[0].size/chunk_size)*chunk_size], [-1, chunk_size])
#
#     split.append(d)
#
# d = np.reshape(split, -1)
# # print(d.shape)
# print(d)
# print(d.shape)
#
# print(np.random.permutation(d))
#
# # c = [i for i in range(15000)]
# # print(c)
# # #
# # # print(b)
# # for i in d:
# #     print([c[j] for j in i])
#
# for i in range(10, 15):
#     pass'

data = [1, 1, 1, 2, 2, 3, 4, 5]
data = np.array(data)
lis = np.unique(data)
id2data = {d: "label" for idx, d in enumerate(lis)}
print(id2data)
for l in lis:
    d = np.where(data == l)
    print(len(d[0]))
    if len(d[0]) < 2:
        id2data.pop(l)
print(id2data)

a = 76
print(a)
print(a // 10)
print(a // 10 * 10)

targets = np.array([1, 1, 1, 1,3, 4, 5, 5])
l_ = 1
l = np.where(targets == l_)
print(l)
print(len(l))


a = np.array([1, 2, 3, 4, 5, 6, 7])
b = torch.tensor(a)
print(b)



