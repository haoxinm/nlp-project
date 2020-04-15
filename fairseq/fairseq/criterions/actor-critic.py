import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_transformers import BertModel, BertTokenizer
from fairseq import metrics, utils
from fairseq.data.encoders.gpt2_bpe import GPT2BPE_modified
from fairseq.models.bart import BARTModel
from . import FairseqCriterion, register_criterion

'''
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
    # del lprobs
    # del target
    # del smooth_loss
    # torch.cuda.empty_cache()

    return loss, nll_loss.item()


def critic_loss(net_input, target, task=None, bpe=None, critic=None, output_tokens=None,
                debug=False, max_tokens=1024):
    if debug:
        sentence_txt = bpe.decode(task.source_dictionary.string(net_input))
        target_txt = bpe.decode(task.target_dictionary.string(target))
        output_txt = bpe.decode(task.target_dictionary.string(output_tokens))
        print("\n\n## sentence_txt: ", sentence_txt,  "\n## target_txt: ", target_txt,
              "\n## output_txt: ", output_txt, )
    ub = output_tokens.size(0)
    zeros = torch.zeros((ub, 1), device=output_tokens.device, dtype=output_tokens.dtype)
    ones = torch.ones((ub, 1), device=output_tokens.device, dtype=output_tokens.dtype)
    txt_length = net_input.size(-1)
    summ_length = output_tokens.size(-1)
    if txt_length > max_tokens - 1 - summ_length:
        net_input = net_input[..., :max_tokens - summ_length - 2]
        net_input = torch.cat([net_input, 2 * ones], dim=-1)
    # else:
    #     input_tk = net_input
    cat_input = torch.cat([zeros, net_input, output_tokens], dim=-1)
    # del net_input, output_tokens
    # torch.cuda.empty_cache()
    # print("")
    # print(net_input)
    # print("")
    # print(input_tk)
    # print("")
    # print(cat_input)
    # print("")
    # print(cat_input.shape)
    prev_output_tokens = cat_input.roll(1)
    # prev_output_tokens = cat_input.clone()
    #
    # prev_output_tokens[:, 0] = cat_input.gather(
    #     1,
    #     (cat_input.ne(task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
    # ).squeeze()
    #
    # prev_output_tokens[:, 1:] = cat_input[:, :-1]
    # print((prev_output_tokens==cat_input.roll(1)).all())

    net_output = critic(src_tokens=cat_input, src_lengths=None,
                        prev_output_tokens=prev_output_tokens,
                        features_only=True, )[0]
    idx = cat_input.eq(task.source_dictionary.eos())
    # del cat_input, prev_output_tokens
    # torch.cuda.empty_cache()
    net_output = net_output[idx, :
                 ].view(net_output.size(0), -1, net_output.size(-1))[:, -1, :]

    net_output = critic.classification_heads['critic'](net_output)
    net_output = F.softmax(net_output, dim=1)
    critic_score = net_output[0, 0].item()
    # print("")
    # print(critic_score)
    # print(net_output.sum())

    if debug:
        print("\n## score :", critic_score)
        print("===" * 10)

    return critic_score


def raw_bert_encoder(model, tokenizer, sent_list, stride=128):
    merged_text = ''
    for ss in sent_list: merged_text += ss+' '
    tokens = tokenizer.encode(merged_text)
    #print(len(tokens))

    model.eval()
    #with torch.no_grad():
    if True:
        if len(tokens) <= 510:
            tokens = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)
            vv = model(tokens)[0][0]
            vv = vv.mean(dim=0)
        else:
            end_pointer = stride
            batch = []
            real_length = []
            att_masks = []
            while True:
                start_pointer = end_pointer-510
                if start_pointer < 0: start_pointer = 0
                if start_pointer >= len(tokens): break
                if end_pointer <= len(tokens):
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(end_pointer-start_pointer)
                    att_masks.append([1]*real_length[-1])
                else:
                    batch.append(tokens[start_pointer:end_pointer])
                    real_length.append(len(tokens)-start_pointer)
                    att_masks.append([1] * real_length[-1])
                end_pointer += stride
                #print(len(batch[-1]))

            #padding
            longest = max(real_length)
            for ii in range(len(batch)):
                batch[ii] += [0] * (longest-real_length[ii])
                att_masks[ii] += [0] * (longest-real_length[ii])

            batch = torch.tensor(batch)
            att_masks = torch.tensor(att_masks)

            last_layers = model(input_ids=batch,attention_mask=att_masks)[0]
            vectors = []
            for ii,bb in enumerate(last_layers):
                vectors.append(bb[:real_length[ii]].mean(axis=0))
            vv = torch.tensor(vectors).mean(dim=0)
    return vv


def build_model(model_type, vec_length, learn_rate=None):
    if 'linear' in model_type:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, 1),
        )
    else:
        deep_model = torch.nn.Sequential(
            torch.nn.Linear(vec_length, int(vec_length/2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(vec_length/2), 1),
        )
    if learn_rate is not None: # Not activated for SenSim
        optimiser = torch.optim.Adam(deep_model.parameters(),lr=learn_rate)
        print("Error! See rewarder.py")
        raise
        return deep_model, optimiser
    else:
        return deep_model


class Rewarder:
    def __init__(self,weight_path,model_type='linear',vec_dim=1024,device='cuda:0'):
        self.device = device
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        # self.bert_tokenizer.half()
        # self.bert_model = BertModel.from_pretrained('bert-large-uncased')
        # self.bert_model.half()
        # self.bert_model.to(device)
        # self.bert_model.eval()
        self.reward_model = build_model(model_type, vec_dim*2) # times 2 because the input to the model is the concatenation of the doc-vec and the summ-vec
        self.reward_model.load_state_dict(torch.load(weight_path))
        self.reward_model.to(device)
        self.reward_model.half()
        self.reward_model.eval()

    def __call__(self, output_vec, target_vec):

        # output_vec = raw_bert_encoder(self.bert_model, self.bert_tokenizer, [output_txt])
        # target_vec = raw_bert_encoder(self.bert_model, self.bert_tokenizer, [target_txt])
        input_vec = torch.cat([target_vec, output_vec], dim=-1).half().to(self.device)
        # print(input_vec.shape)
        pred_score = self.reward_model(input_vec).view(1,-1)[0][0]
        return pred_score.item()
'''

class Padder(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def forward(self, input):
        # print(input.shape)
        length = input.size(-1)
        input = F.pad(input, (0, self.max_length - length), mode='constant', value=0)
        # print(input.shape)
        return input


@register_criterion('ac_loss_actor')
class ActorCriterion(FairseqCriterion):

    def __init__(self, task, encoder_json, vocab_bpe, critic_path, critic_file, sentence_avg, max_tokens,
                 print_update, critic_weight, label_smoothing, use_reward, rewarder_file, rewarder_weight):
        super().__init__(task)
        self.eps = label_smoothing
        # self.task = task
        self.debugCount = 0
        self.bpe = GPT2BPE_modified(encoder_json, vocab_bpe)
        self.sentence_avg = sentence_avg
        self.max_tokens = max_tokens
        self.print_update = print_update
        self.critic_weight = critic_weight
        # print(critic_path)
        self.critic = BARTModel.from_pretrained(critic_path, checkpoint_file=critic_file).model
        self.critic.half()
        # self.critic = self.critic.cpu()
        # self.critic.short()
        self.critic.eval()
        self.use_rewarder = use_reward
        self.rewarder_weight = rewarder_weight
        if use_reward:
            self.padder = Padder(1024)
            # self.rewarder = Rewarder(rewarder_file)

    # '''
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--critic-path', default='checkpoints/critic', metavar='DIR',
                            help='path to load critic')
        parser.add_argument('--critic-file', default='model.pt', metavar='DIR',
                            help='file to load critic')
        parser.add_argument('--encoder-json', default='encoder.json', metavar='DIR',
                            help='file to load encoder.json')
        parser.add_argument('--vocab-bpe', default='vocab.bpe', metavar='DIR',
                            help='file to load vocab.bpe')
        parser.add_argument('--critic-weight', default=100., type=float,
                            help='weight for critic score')
        parser.add_argument('--print-update', default=10, type=int,
                            help='number of updates before printing the network output for check')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--use-reward', action='store_true',
                            help='whether to use rewarder')
        parser.add_argument('--rewarder-file', default='pretrained/sample.model', metavar='DIR',
                            help='file to load rewarder')
        parser.add_argument('--rewarder-weight', default=500., type=float,
                            help='weight for reward')
        # fmt: on
        # '''

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample['net_input'])[0]
        # print(net_output)
        # loss = 0.
        net_output = utils.log_softmax(net_output, dim=-1)
        # device = net_output.device
        net_output = net_output
        # print(net_output.device)
        target = sample['target']
        t_shape = target.shape
        src_tokens = sample['net_input']['src_tokens']
        ntokens = sample['ntokens']
        # del sample, model
        # torch.cuda.empty_cache()
        # critic_score = self.compute_critic_loss(torch.argmax(net_output, -1), target, src_tokens)
        # loss, nll_loss = self.compute_loss(net_output, target, reduce=reduce)
        # compute label smoothed cross entropy
        target = target.view(-1, 1)
        lprobs = net_output.view(-1, net_output.size(-1))
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.padding_idx is not None:
            pad_mask = target.eq(self.padding_idx)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        # del lprobs, smooth_loss
        # torch.cuda.empty_cache()

        # compute critic score
        target = target.view(t_shape)
        net_output = torch.argmax(net_output, -1)
        debug = False
        self.debugCount += 1
        if self.print_update != 0 and self.debugCount % self.print_update == 0:
            debug = True

        if debug:
            sentence_txt = self.bpe.decode(self.task.source_dictionary.string(src_tokens))
            target_txt = self.bpe.decode(self.task.target_dictionary.string(target))
            output_txt = self.bpe.decode(self.task.target_dictionary.string(net_output))
            print("\n\n## sentence_txt: ", sentence_txt, "\n## target_txt: ", target_txt,
                  "\n## output_txt: ", output_txt, )

        reward = 0.
        if self.use_rewarder:
        #     target_txt = self.bpe.decode(self.task.target_dictionary.string(target))
        #     output_txt = self.bpe.decode(self.task.target_dictionary.string(net_output))
            reward = self.rewarder(self.padder(net_output), self.padder(target))

        ub = net_output.size(0)
        zeros = torch.zeros((ub, 1), dtype=net_output.dtype).cuda()
        ones = torch.ones((ub, 1), dtype=net_output.dtype).cuda()
        txt_length = src_tokens.size(-1)
        summ_length = net_output.size(-1)
        if txt_length > self.max_tokens - 1 - summ_length:
            src_tokens = src_tokens[..., :self.max_tokens - summ_length - 2]
            src_tokens = torch.cat([src_tokens, 2 * ones], dim=-1)
        cat_input = torch.cat([zeros, src_tokens, net_output], dim=-1)
        # print(device)
        # cat_input = cat_input.to('cpu')
        # print(cat_input.device)
        prev_output_tokens = cat_input.roll(1)

        with torch.no_grad():
            net_output = self.critic(src_tokens=cat_input, src_lengths=None,
                                prev_output_tokens=prev_output_tokens,
                                features_only=True, )[0]
        idx = cat_input.eq(self.task.source_dictionary.eos())
        # del cat_input, prev_output_tokens
        # torch.cuda.empty_cache()
        net_output = net_output[idx, :
                     ].view(net_output.size(0), -1, net_output.size(-1))[:, -1, :]

        net_output = self.critic.classification_heads['critic'](net_output)
        net_output = utils.log_softmax(net_output, dim=1)
        # print(net_output)
        critic_score = torch.exp(net_output[0, 1]).item()
        # print(critic_score)

        if debug:
            print("\n## critic score :", critic_score)
            if self.use_rewarder:
                print("\n## reward :", reward)
            print("===" * 10)

        loss = loss + self.critic_weight * critic_score
        if self.use_rewarder:
            loss = loss - self.rewarder_weight * reward
        sample_size = target.size(0) if self.sentence_avg else ntokens
        logging_output = {
            'loss': loss.item(),
            'nll_loss': nll_loss.item(),
            'critic_score': critic_score,
            'ntokens': ntokens,
            'nsentences': target.size(0),
            'sample_size': sample_size,
        }
        if self.use_rewarder:
            logging_output['reward'] = reward
        # del critic_score, nll_loss
        # torch.cuda.empty_cache()
        return loss, sample_size, logging_output
    '''
    def compute_loss(self, net_output, target, reduce=True):
        # device = net_output
        # net_output = utils.log_softmax(net_output, dim=-1).half()
        net_output = net_output.view(-1, net_output.size(-1))
        target = target.view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(net_output, target, self.eps,
                                                 ignore_index=self.padding_idx,
                                                 reduce=reduce)

        return loss, nll_loss

    def compute_critic_loss(self, net_output, target, net_input):
        debug = False
        self.debugCount += 1
        if self.print_update != 0 and self.debugCount % self.print_update == 0:
            debug = True
        # print("")
        # print(sample['net_input']['src_tokens'].shape)
        # print("")
        # print(sample['net_input'])
        critic_score = \
            critic_loss(net_input=net_input, target=target,
                        task=self.task, bpe=self.bpe, critic=self.critic,
                        output_tokens=net_output,
                        debug=debug, max_tokens=self.max_tokens,
                        )

        return critic_score
    '''
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        critic_score_sum = sum(log.get('critic_score', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('critic_score', critic_score_sum / sample_size / math.log(2), sample_size)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        if 'reward' in logging_outputs[0].keys():
            reward_sum = sum(log.get('reward', 0) for log in logging_outputs)
            metrics.log_scalar('reward', reward_sum / sample_size / math.log(2), sample_size, round=3)


@register_criterion('ac_loss_critic')
class CriticCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, actor_path,
                 actor_file, max_tokens):
        super().__init__(task)
        # self.eps = label_smoothing
        # self.task = task
        # self.debugCount = 0
        # args.bpe = 'gpt2'
        # self.bpe = encoders.build_bpe(args)
        # print(args.actor_path)
        self.regression_target = False
        self.classification_head_name = classification_head_name
        self.actor = BARTModel.from_pretrained(actor_path, checkpoint_file=actor_file).model
        # self.actor.half()
        self.actor.eval()
        self.max_tokens = max_tokens
        # self.loss_weight = args.loss_weight

    # '''
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--actor-path', default='checkpoints/actor', metavar='DIR',
                            help='path to load actor')
        parser.add_argument('--actor-file', default='model.pt', metavar='DIR',
                            help='file to load actor')
        # fmt: on
        # '''

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # print(sample['net_input'].keys())
        sample['net_input']['prev_output_tokens'] = sample['target'].roll(1, dims=-1)
        with torch.no_grad():
            actor_output = self.actor(**sample['net_input'])
        # """
        if np.random.random() < 0.5:
            # use target
            output_tk = model.get_targets(sample, actor_output)
            ub = output_tk.size(0)
            zeros = torch.zeros((ub, 1), dtype=output_tk.dtype).cuda()
            ones = torch.ones((ub, 1), dtype=output_tk.dtype).cuda()
            new_target = zeros
        else:
            # use network output
            output_tk = torch.argmax(utils.log_softmax(actor_output[0], dim=-1), -1)
            ub = output_tk.size(0)
            zeros = torch.zeros((ub, 1), dtype=output_tk.dtype).cuda()
            ones = torch.ones((ub, 1), dtype=output_tk.dtype).cuda()
            new_target = ones
        '''
        actor_output_tk = torch.argmax(utils.log_softmax(actor_output[0], dim=-1), -1)
        target = model.get_targets(sample, actor_output)

        ub = output_tk.size(0)
        ones = torch.ones((ub, 1), device=actor_output_tk.device, dtype=actor_output_tk.dtype)
        zeros = torch.zeros((ub, 1), device=actor_output_tk.device, dtype=actor_output_tk.dtype)
        # '''

        input_tk = sample['net_input']['src_tokens']

        # truncate long inputs
        txt_length = input_tk.size(-1)
        summ_length = output_tk.size(-1)

        if txt_length > self.max_tokens - 1 - summ_length:
            input_tk = input_tk[..., :self.max_tokens - summ_length - 2]
            input_tk = torch.cat([input_tk, 2 * ones], dim=-1)

        new_src_tokens = torch.cat([zeros, input_tk, output_tk], dim=-1)
        new_src_length = new_src_tokens.size(-1)

        idx = torch.randperm(new_src_tokens.size(0))
        new_src_tokens = new_src_tokens[idx]
        new_target = new_target[idx]
        new_prev_output_tokens = new_src_tokens.roll(1, dims=-1)

        sample['net_input'] = {'src_tokens': new_src_tokens,
                               'src_lengths': new_src_length,
                               'prev_output_tokens': new_prev_output_tokens}
        sample['target'] = new_target
        # loss, sample_size, logits, targets = self.compute_loss(sample, model)
        logits, _ = model(**sample['net_input'],
                          features_only=True,
                          classification_head_name=self.classification_head_name)
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')
        # '''
        # """
        '''
        input_tk = sample['net_input']['src_tokens']
        actor_output_tk = torch.argmax(utils.log_softmax(actor_output[0], dim=-1), -1)
        target = model.get_targets(sample, actor_output)

        ub = actor_output_tk.size(0)
        ones = torch.ones((ub, 1), device=actor_output_tk.device, dtype=actor_output_tk.dtype)
        zeros = torch.zeros((ub, 1), device=actor_output_tk.device, dtype=actor_output_tk.dtype)

        txt_length = input_tk.size(-1)
        summ_length = max(actor_output_tk.size(-1), target.size(-1))

        if txt_length > self.max_tokens - 1 - summ_length:
            input_tk = input_tk[..., :self.max_tokens - summ_length - 2]
            input_tk = torch.cat([input_tk, 2 * ones], dim=-1)

        new_input_hypo = torch.cat([zeros, input_tk, actor_output_tk], dim=-1)
        new_input_tg = torch.cat([zeros, input_tk, target], dim=-1)
        new_src_tokens = torch.cat([new_input_hypo, new_input_tg], dim=0)
        new_target = torch.cat([zeros, ones], dim=0)
        new_src_length = new_src_tokens.size(-1)

        idx = torch.randperm(new_src_tokens.size(0))
        new_src_tokens = new_src_tokens[idx]
        new_target = new_target[idx]
        new_prev_output_tokens = new_src_tokens.roll(1, dims=-1)

        sample['net_input'] = {'src_tokens': new_src_tokens[:ub, ...],
                               'src_lengths': new_src_length,
                               'prev_output_tokens': new_prev_output_tokens[:ub, ...]}
        sample['target'] = new_target[:ub, ...]
        loss1, sample_size1, logits1, target1 = self.compute_loss(sample, model)
        sample['net_input'] = {'src_tokens': new_src_tokens[ub:, ...],
                               'src_lengths': new_src_length,
                               'prev_output_tokens': new_prev_output_tokens[ub:, ...]}
        sample['target'] = new_target[ub:, ...]
        loss2, sample_size2, logits2, target2 = self.compute_loss(sample, model)

        loss = loss1 + loss2
        sample_size = sample_size1 + sample_size2
        logits = torch.cat([logits1, logits2], dim=0)
        targets = torch.cat([target1, target2], dim=0)
        # '''

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()
        # torch.cuda.empty_cache()

        return loss, sample_size, logging_output

    def compute_loss(self, sample, model):
        logits, _ = model(**sample['net_input'],
                          features_only=True,
                          classification_head_name=self.classification_head_name)
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')

        # torch.cuda.empty_cache()

        return loss, sample_size, logits, targets

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
