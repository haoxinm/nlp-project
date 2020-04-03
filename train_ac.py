#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.trainer import Trainer
from fairseq.model_parallel.megatron_trainer import MegatronTrainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        # checkpoint_utils.verify_checkpoint_directory(args.save_dir)
        checkpoint_utils.verify_checkpoint_directory(args.actor_path)
        checkpoint_utils.verify_checkpoint_directory(args.critic_path)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    # task = tasks.setup_task(args)
    actor_args, actor_task, actor_model, actor_criterion, actor_trainer, \
    actor_epoch_itr, actor_extra_state = get_ready(args, 'a')

    critic_args, critic_task, critic_model, critic_criterion, critic_trainer, \
    critic_epoch_itr, critic_extra_state = get_ready(args, 'a')

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        actor_task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    '''
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    '''

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    actor_lr = actor_trainer.get_lr()
    critic_lr = critic_trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while (
        min(actor_lr, critic_lr) > args.min_lr
        and max(actor_epoch_itr.next_epoch_idx, critic_epoch_itr.next_epoch_idx) <= max_epoch
        and max(actor_trainer.get_num_updates(), critic_trainer.get_num_updates()) < max_update
    ):
        # train for one epoch
        actor_args, actor_trainer, actor_task, actor_epoch_itr, \
        critic_args, critic_trainer, critic_task, critic_epoch_itr \
            = train_ac(actor_args, actor_trainer, actor_task, actor_epoch_itr,
                       critic_args, critic_trainer, critic_task, critic_epoch_itr)

        if not args.disable_validation and actor_epoch_itr.epoch % args.validate_interval == 0:
            actor_valid_losses = validate(actor_args, actor_trainer, actor_task, actor_epoch_itr, valid_subsets)
        else:
            actor_valid_losses = [None]
        if not args.disable_validation and critic_epoch_itr.epoch % args.validate_interval == 0:
            critic_valid_losses = validate(critic_args, critic_trainer, critic_task, critic_epoch_itr, valid_subsets)
        else:
            critic_valid_losses = [None]

        # only use first validation loss to update the learning rate
        actor_lr = actor_trainer.lr_step(actor_epoch_itr.epoch, actor_valid_losses[0])
        critic_lr = critic_trainer.lr_step(critic_epoch_itr.epoch, critic_valid_losses[0])

        # save checkpoint
        if actor_epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(actor_args, actor_trainer, actor_epoch_itr, actor_valid_losses[0])
        if critic_epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(critic_args, critic_trainer, critic_epoch_itr, critic_valid_losses[0])

        # early stop
        if should_stop_early(args, actor_valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        actor_epoch_itr = actor_trainer.get_train_iterator(
            actor_epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
        critic_epoch_itr = critic_trainer.get_train_iterator(
            critic_epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def get_ready(args, ac='a'):
    train_args = deepcopy(args)
    if ac=='a':
        train_args.restore_file = args.actor_restore_file
        train_args.task = args.actor_task
        train_args.criterion = args.actor_criterion
        train_args.save_interval_updates = args.actor_save_update
    elif ac=='c':
        train_args.restore_file = args.critic_restore_file
        train_args.task = args.critic_task
        train_args.criterion = args.critic_criterion
        train_args.save_interval_updates = args.critic_save_update
    task = tasks.setup_task(train_args)
    model = task.build_model(train_args)
    criterion = task.build_criterion(train_args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(train_args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    if train_args.model_parallel_size == 1:
        trainer = Trainer(train_args, task, model, criterion)
    else:
        trainer = MegatronTrainer(train_args, task, model, criterion)

    logger.info('training on {} GPUs'.format(train_args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        train_args.max_tokens,
        train_args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(train_args, trainer)

    return train_args, task, model, criterion, trainer, epoch_itr, extra_state


def train_ac(actor_args, actor_trainer, actor_task, actor_epoch_itr,
             critic_args, critic_trainer, critic_task, critic_epoch_itr):
    itr = actor_epoch_itr.next_actor_epoch_itr(
        fix_batches_to_gpus=actor_args.fix_batches_to_gpus,
        shuffle=(actor_epoch_itr.next_epoch_idx > actor_args.curriculum),
    )
    update_freq = (
        actor_args.update_freq[actor_epoch_itr.epoch - 1]
        if actor_epoch_itr.epoch <= len(actor_args.update_freq)
        else actor_args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=actor_args.log_format,
        log_interval=actor_args.log_interval,
        epoch=actor_epoch_itr.epoch,
        tensorboard_logdir=(
            actor_args.tensorboard_logdir if distributed_utils.is_master(actor_args) else None
        ),
        default_log_format=('tqdm' if not actor_args.no_progress_bar else 'simple'),
    )

    # actor_task specific setup per epoch
    actor_task.begin_epoch(actor_epoch_itr.epoch, actor_trainer.get_model())

    valid_subsets = actor_args.valid_subset.split(',')
    # max_update = actor_args.max_update or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = actor_trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = actor_trainer.get_num_updates()
        if num_updates % actor_args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
                not actor_args.disable_validation
                and actor_args.save_interval_updates > 0
                and num_updates % actor_args.save_interval_updates == 0
                and num_updates > 0
        ):
            valid_losses = validate(actor_args, actor_trainer, actor_task, actor_epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(actor_args, actor_trainer, actor_epoch_itr, valid_losses[0])

            critic_args, critic_trainer, critic_task, critic_epoch_itr \
                = train_critic(critic_args, critic_trainer, critic_task, critic_epoch_itr)
    return actor_args, actor_trainer, actor_task, actor_epoch_itr, \
           critic_args, critic_trainer, critic_task, critic_epoch_itr


def train_critic(args, trainer, task, epoch_itr):
    print("...... training critic ......")
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    # max_update = args.max_update or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
                not args.disable_validation
                and args.save_interval_updates > 0
                and num_updates % args.save_interval_updates == 0
                and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
            break

    print("...... training actor ......")

    return args, trainer, task, epoch_itr


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs >= args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
