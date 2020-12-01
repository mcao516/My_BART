#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import logging
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from fairseq.tasks.translation import TranslationTask
from fairseq.data.language_pair_dataset import collate

from modules.data_utils import FairseqDataset
from modules.trainer import Trainer
from modules.utils import init_arg_parser


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq.train')


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_dictionary(path, src_dict_name='source', tgt_dict_name='target'):
    """Load source & target fairseq dictionary.
    """
    # path = self.args.data_name_or_path
    src_dict = TranslationTask.load_dictionary(os.path.join(path, 'dict.{}.txt'.format(src_dict_name)))
    tgt_dict = TranslationTask.load_dictionary(os.path.join(path, 'dict.{}.txt'.format(tgt_dict_name)))

    assert src_dict.bos() == tgt_dict.bos() == 0
    assert src_dict.pad() == tgt_dict.pad() == 1
    assert src_dict.eos() == tgt_dict.eos() == 2
    assert src_dict.unk() == tgt_dict.unk() == 3
    logger.info('[{}] dictionary: {} types'.format('source', len(src_dict)))
    logger.info('[{}] dictionary: {} types'.format('target', len(tgt_dict)))

    return src_dict, tgt_dict


def main(rank, args, world_size):
    if rank == 0:
        logger.info(args)

    # create task & load source and taget dictionary
    # translation_task = TranslationTask.setup_task(args)

    logger.info(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # build trainer
    logger.info('- build trainer (rank {})...'.format(rank))
    trainer = Trainer(args, logger, rank)

    src_dict, tgt_dict = trainer.get_dicts()

    # create datasets
    logger.info('- loading training set (rank {})...'.format(rank))
    train_dataset = FairseqDataset(src_dict, args.train_source, args.train_target,
                                   max_positions=args.max_positions, no_bos=args.no_bos)
    
    logger.info('- loading development set (rank {})...'.format(rank))
    dev_dataset = FairseqDataset(src_dict, args.dev_source, args.dev_target,
                                 max_positions=args.max_positions, no_bos=False)

    torch.distributed.barrier() # make sure all datasets are loaded

    def collate_fn(samples):
        """
        Args:
            samples: list of samples
        """
        return collate(samples, train_dataset.pad_idx, train_dataset.eos_idx, 
                       left_pad_source=True,
                       left_pad_target=False,
                       input_feeding=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                  collate_fn=collate_fn, pin_memory=True)

    # train model
    trainer.train(train_dataloader, train_sampler, dev_dataset, None)

    # finish process
    cleanup()


if __name__ == "__main__":
    parser = init_arg_parser()
    # TranslationTask.add_args(parser)

    args = parser.parse_args()

    # main(args)
    n_gpus = torch.cuda.device_count()
    mp.spawn(main,
             args=(args, n_gpus),
             nprocs=n_gpus,
             join=True)