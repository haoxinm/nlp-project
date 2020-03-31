import math
import torch

from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTModel

from . import FairseqCriterion, register_criterion
from fairseq.tasks import TASK_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY


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


@register_criterion('ac-loss-actor')
class ActorCriterion(FairseqCriterion):

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
        self.critic = BARTModel.from_pretrained(args.critic_path, checkpoint_file='model.pt')
        self.loss_weight = args.loss_weight

    # '''
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--actor-task', default='translation', metavar='TASK',
                            choices=TASK_REGISTRY.keys(), help='task for actor')
        parser.add_argument('--actor-criterion', default='ac-loss-actor', metavar='CRITERION',
                            choices=CRITERION_REGISTRY.keys(), help='criterion for actor')
        parser.add_argument('--actor-save-update', '--asu', default=0, type=int, metavar='N',
                           help='force stop training actor at specified update')
        parser.add_argument('--critic-path', default='checkpoints/critic', metavar='DIR',
                            help='path to load/save critic')
        # fmt: on
        # '''

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


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('ac-loss-critic')
class CriticCriterion(FairseqCriterion):

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
        self.actor = BARTModel.from_pretrained(args.actor_path, checkpoint_file='model.pt')
        self.loss_weight = args.loss_weight

    # '''
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--critic-task', default='translation', metavar='TASK',
                            choices=TASK_REGISTRY.keys(), help='task for critic')
        parser.add_argument('--critic-criterion', default='ac-loss-critic', metavar='CRITERION',
                            choices=CRITERION_REGISTRY.keys(), help='criterion for critic')
        parser.add_argument('--critic-save-update', '--csu', default=0, type=int, metavar='N',
                            help='force stop training critic at specified update')
        parser.add_argument('--actor-path', default='checkpoints/actor', metavar='DIR',
                            help='path to load/save actor')
        # fmt: on
        # '''

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        actor_output = self.actor(**sample['net_input'])
        actor_output_tk = actor_output.argmax(-1)
        ub = sample['target'].size(0)
        target = model.get_targets(sample, actor_output).view(-1, 1)
        zeros = torch.zeros((ub, 1))
        ones = torch.ones((ub, 1))
        input_tk = sample['net_input']['src_tokens']
        new_input_hypo = torch.cat([zeros, input_tk, actor_output_tk], dim=-1)
        new_input_tg = torch.cat([zeros, input_tk, target], dim=-1)
        new_src_tokens = torch.cat([new_input_hypo, new_input_tg], dim=0)
        new_target = torch.cat([zeros, ones], dim=0)
        new_src_length = new_src_tokens.size(-1)
        new_prev_output_tokens = new_src_tokens.roll(1, dims=-1)
        sample['net_input'] = {'src_tokens': new_src_tokens, 'src_length': new_src_length,
                               'prev_output_tokens': new_prev_output_tokens}
        sample['target'] = new_target
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            # 'semsim_score': utils.item(semsim_score) if reduce else semsim_score,  # semsim_score : int
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    # SWEM
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # print(lprobs.shape)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        # print(lprobs.shape)
        # print(**sample['net_input'].shape)
        print(sample['net_input'])
        target = model.get_targets(sample, net_output).view(-1, 1)
        # print("")
        # print(target.shape)
        # print("")
        # print(target)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

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
