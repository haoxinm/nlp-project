import torch
from fairseq.models.bart import BARTModel
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--checkpoint-path', default='checkpoints/actor', metavar='DIR',
                            help='path to load checkpoint')
parser.add_argument('--checkpoint-file', default='checkpoint_best.pt', metavar='DIR',
                            help='file to load actor')
parser.add_argument('--dataset', default='cnn_dm', metavar='DIR',
                            help='dataset to evaluate')
parser.add_argument('--batch-size', default=50, metavar='N', type=int,
                            help='dataset to evaluate')
parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')

args, _ = parser.parse_known_args()

bart = BARTModel.from_pretrained(
    args.checkpoint_path,
    checkpoint_file=args.checkpoint_file,
    data_name_or_path=args.dataset+'-bin'
)

if args.cpu:
    bart.cpu()
else:
    bart.cuda()
    bart.half()

bart.eval()
# if torch.cuda.device_count() > 1:
#     bart.model = torch.nn.DataParallel(bart.model)
count = 1
bsz = args.batch_size
num_lines = sum(1 for _ in open(args.dataset+'/test.source'))
with open(args.dataset+'/test.source') as source, open(args.dataset+'/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in tqdm(source, total=num_lines):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()