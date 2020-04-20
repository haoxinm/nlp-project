# import nltk.data
# from nltk import tokenize

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
import numpy as np


basedir = './data'
train_output_text_file = open("short_story/train.source", "w",encoding="utf-8")
train_output_summary_file = open("short_story/train.target", "w",encoding="utf-8")
val_output_text_file = open("short_story/val.source", "w",encoding="utf-8")
val_output_summary_file = open("short_story/val.target", "w",encoding="utf-8")
test_output_text_file = open("short_story/test.source", "w",encoding="utf-8")
test_output_summary_file = open("short_story/test.target", "w",encoding="utf-8")
train_count= 0
val_count=0
test_count=0
subdir1_list = os.listdir(basedir)
for subdir1_enum in subdir1_list:
    subdir2 = basedir+ '/'+ subdir1_enum
    subdir2_list = os.listdir(subdir2)
    for subdir2_enum in subdir2_list:
        subdir3 = subdir2 + '/'+ subdir2_enum
        subdir3_list = os.listdir(subdir3)
        for subdir4_enum in subdir3_list:
            subdir4 = subdir3 + '/' + subdir4_enum
            f_text = open(subdir4 + "/text.txt", "r", encoding="utf-8")
            f_summary = open(subdir4 + "/summary.txt", "r", encoding="utf-8")
            text_data = f_text.read()
            summary_data = f_summary.read()

            # sent_list_text = tokenize.sent_tokenize(text_data)
            # sent_list_summary = tokenize.sent_tokenize(summary_data)
            output_text_string = text_data.replace('\n', ' ')
            output_summary_string = summary_data.replace('\n', ' ')
            for item in ["\u201f","\u2011"]:
                output_text_string = output_text_string.replace(item, "-")
                output_summary_string = output_summary_string.replace(item, "-")
            # for item in sent_list_text:
            #     item=item.replace('\n',' ')
            #     output_text_string+=item
            # for item in sent_list_summary:
            #     item=item.replace('\n',' ')
            #     output_summary_string+=item
            i=np.random.random()
            if i < 0.8:
                train_count += 1
                train_output_text_file.writelines(output_text_string + '\n')
                train_output_summary_file.writelines(output_summary_string + '\n')
            elif i>=0.8 and i<0.9:
                val_count += 1
                val_output_text_file.writelines(output_text_string + '\n')
                val_output_summary_file.writelines(output_summary_string + '\n')
            else:
                test_count += 1
                test_output_text_file.writelines(output_text_string + '\n')
                test_output_summary_file.writelines(output_summary_string + '\n')

train_output_text_file.close()
train_output_summary_file.close()
val_output_text_file.close()
val_output_summary_file.close()
test_output_text_file.close()
test_output_summary_file.close()

print("train_count:"+str(train_count))
print("val_count:"+str(val_count))
print("test_count:"+str(test_count))