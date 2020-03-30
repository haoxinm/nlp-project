import math
import torch

from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTModel

from . import FairseqCriterion, register_criterion


def critic_loss(input_tokens, task=None, bpe=None, critic=None, output_tokens=None, ignore_index=None,
                debug=False):
    target = torch.argmax(utils.log_softmax(output_tokens, dim=-1), -1)  # maxpool

    if ignore_index is not None:
        non_pad_mask = input_tokens.ne(ignore_index)
        sentence = input_tokens[non_pad_mask]
    else:
        sentence = input_tokens

    critic_score = critic(input_tokens, output_tokens)
    if debug:
        sentence_txt = bpe.decode(task.source_dictionary.string(sentence))
        target_txt = bpe.decode(task.target_dictionary.string(target))
        print("\n\n## sentence_txt: ", sentence_txt, "\n## target_txt: ", target_txt, "\n## Reward :", critic_score)

    if debug:
        print("===" * 10)
    return critic_score  # semsim_score : semsim_score


@register_criterion('ac_loss')
class ActorCriticCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.task = task
        self.debugCount = 0
        args.bpe = 'gpt2'
        self.bpe = encoders.build_bpe(args)
        """
        if args.rewarderpath == None:
            args.rewarderpath = "./semsim/trained_models/" + args.restore_file.split('/')[-1] # TODO : refactoring required
            print("args.rewarderpath not set : use %s instead."%args.rewarderpath) 
        """
        self.critic = BARTModel.from_pretrained(args.criticpath, checkpoint_file='model.pt')
        self.loss_weight = args.loss_weight

    '''
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
        '''

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'])
        loss = self.compute_loss(net_output, sample)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            # 'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            # 'semsim_score': utils.item(semsim_score) if reduce else semsim_score,  # semsim_score : int
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    # SWEM
    def compute_loss(self, net_output, sample):
        debug = False
        self.debugCount += 1
        if self.debugCount % 500 == 1:
            debug = False

        loss = critic_loss(sample['net_input'], task=self.task, bpe=self.bpe, critic=self.critic,
                           output_tokens=net_output[0], ignore_index=self.padding_idx,
                           debug=debug
                           )

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(
                2) if sample_size > 0 else 0.,
            # 'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
            #     2) if ntokens > 0 else 0.,
            # 'semsim_score': sum(log.get('semsim_score', 0) for log in logging_outputs) / sample_size / math.log(
            #     2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
