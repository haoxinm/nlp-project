import nltk.data
from nltk import tokenize

# # basedir = './Authors/Alice Munro/Mrs. Cross and Mrs. Kidd/'
# basedir = './'
# # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# fp = open(basedir+"summary.txt","r")
# data = fp.read()
# # print('\n-----\n'.join(tokenizer.tokenize(data)))
# print("************************************************")
# output = tokenize.sent_tokenize(data)
# print(output)
# print(len(output))



import glob
import pdb
import os


basedir = './data'

# output = []
subdir1_list = os.listdir(basedir)
count = 0
sum = 0
for subdir1_enum in subdir1_list:
    subdir2 = basedir+ '/'+ subdir1_enum
    subdir2_list = os.listdir(subdir2)
    for subdir2_enum in subdir2_list:
        subdir3 = subdir2 + '/'+ subdir2_enum
        subdir3_list = os.listdir(subdir3)
        for subdir4_enum in subdir3_list:
            subdir4 = subdir3 + '/' + subdir4_enum
            fp = open(subdir4 + "/text.txt", "r", encoding="utf8")
            data = fp.read()
            # print('\n-----\n'.join(tokenizer.tokenize(data)))
            print("************************************************")
            count += 1
            output = tokenize.sent_tokenize(data)

            print(subdir4)
            sum += len(output)
            print(count, '  ', sum)
            # print(output)
            # print(len(output))



# print(output)
print(sum)
print("avg=",sum/count)
