import fairseq
import torch
from fairseq.models.bart import BARTModel
from tqdm import tqdm


def get_summary(model, source_file, target_file, new_source):
    model.cuda()
    model.eval()
    model.half()
    count = 1
    bsz = 32
    num_lines = sum(1 for _ in open(source_file))
    with open(source_file) as source, open(target_file, 'w') as fout, open(new_source, 'w') as sout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source, total=num_lines):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = model.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                for s in slines:
                    sout.write(s + '\n')
                    s.flush()
                slines = []
            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = model.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()


def generate_data(model, source_path, target_path):
    for source in ['train', 'valid']:
        print('generate summary for ' + source + '...')
        source_file = source_path + '/' + source + '.source'
        target_file = target_path + '/' + source + '.hypo'
        new_source = source_path + '/' + source + '.new_source'
        get_summary(model, source_file, target_file, new_source)


if __name__ == '__main__':
    bart = BARTModel.from_pretrained('pretrained/BART/bart.large.mnli/', checkpoint_file='model.pt')
    generate_data(bart, 'cnn_dm/', 'cnn_dm/')
