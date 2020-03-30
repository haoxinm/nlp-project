import fairseq
import torch
from fairseq.models.bart import BARTModel


def get_summary(model, source_file, target_file):
    model.cuda()
    model.eval()
    model.half()
    count = 1
    bsz = 32
    with open(source_file) as source, open(target_file, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = model.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = model.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()


def generate_data(source_path, target_path):
    bart = BARTModel.from_pretrained('', checkpoint_file='model.pt')
    for source in ['train, val']:
        source_file = source_path + '/' + source
        target_file = target_path + '/' + source
        get_summary(bart, source_file, target_file)


if __name__ == '__main__':

