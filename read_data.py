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
output_text_file = open("short_story/val.source", "w",encoding="utf-8")
output_summary_file = open("short_story/val.target", "w",encoding="utf-8")
count_text= 0
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
            count_text += 1
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
            output_text_file.writelines(output_text_string + '\n')
            output_summary_file.writelines(output_summary_string + '\n')
output_text_file.close()
output_summary_file.close()

print(count_text)
