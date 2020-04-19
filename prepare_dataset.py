import csv
from tqdm import tqdm
import numpy as np


# def spilt_tsv(source_file, target_file, hypo_file, output_file):
#     num_lines = sum(1 for _ in open(source_file))
#     with open(source_file) as source, open(target_file) as target, \
#             open(hypo_file) as hypo, open(output_file + '.source', 'w') as i0, \
#             open(output_file + '.target', 'w') as i1, open(output_file + '.label', 'w') as label:
#         i0_writer = csv.writer(i0, delimiter='\t')
#         i1_writer = csv.writer(i1, delimiter='\t')
#         # label_writer = csv.writer(label, delimiter='\t')
#         for _ in tqdm(range(num_lines)):
#             if np.random.random() < 0.75:
#                 continue
#             i0_line = source.readline().strip()
#             target_line = target.readline().strip()
#             # hypo_line = hypo.readline().strip()
#             # i0_writer.writerow([i0_line])
#             # i1_writer.writerow([hypo_line])
#             # label_writer.writerow([0])
#             i0_writer.writerow([i0_line])
#             i1_writer.writerow([target_line])
#             # label_writer.writerow([1])

# if __name__ == '__main__':
#     base_path = 'cnn_dm'
#     for source in ['/val']:
#         source_file = base_path + source + '.source'
#         target_file = base_path + source + '.target'
#         hypo_file = base_path + source + '.hypo'
#         output_file = base_path + source + '_quarter'
#         spilt_tsv(source_file, target_file, hypo_file, output_file)

def spilt_tsv(source_file, target_file, hypo_file, output_file):
    num_lines = sum(1 for _ in open(source_file,encoding="utf-8"))
    with open(source_file,"r", encoding="utf-8") as source, \
            open(target_file,"r", encoding="utf-8") as target, \
            open(output_file + '.source', 'w') as i0, \
            open(output_file + '.target', 'w') as i1, \
            open(output_file + '.label', 'w') as label:
        i0_writer = csv.writer(i0, delimiter='\t')
        i1_writer = csv.writer(i1, delimiter='\t')
        # label_writer = csv.writer(label, delimiter='\t')
        for _ in tqdm(range(num_lines)):
            if np.random.random() < 0.75:
                continue
            i0_line = source.readline().strip()
            target_line = target.readline().strip()
            # hypo_line = hypo.readline().strip()
            # i0_writer.writerow([i0_line])
            # i1_writer.writerow([hypo_line])
            # label_writer.writerow([0])
            i0_writer.writerow([i0_line])
            i1_writer.writerow([target_line])
            # label_writer.writerow([1])
if __name__ == '__main__':
    base_path = 'short_story/'
    for source in ['val']:
        source_file = base_path + source + '.source'
        target_file = base_path + source + '.target'
        hypo_file = base_path + source + '.hypo'
        output_file = base_path + source + '_quarter'
        spilt_tsv(source_file, target_file, hypo_file, output_file)
