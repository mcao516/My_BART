#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import logging

from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from fairseq.models.bart import BARTModel
from fairseq.optim.adam import FairseqAdam
from fairseq.utils import move_to_cuda
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule

from modules.criterion import LabelSmoothedCrossEntropyCriterion
from modules.utils import Progbar
from modules.model import BARTWrapper


class Trainer(object):
    """Class for BART model training, evaluation and test."""

    def __init__(self, args, logger, rank):
        super(Trainer, self).__init__()
        self.args = args
        self.logger = logger
        self.rank = rank

        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self._num_updates = 0

        # set cuda device
        self.cuda = torch.cuda.is_available() and not args.cpu
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load BART model
        self._load_bart_model()
        self.bart = self.bart.to(rank)

        # Load source & target dictionary
        self.task = self.bart.task
        self.src_dict, self.tgt_dict = self.bart.task.src_dict, self.bart.task.tgt_dict

        # build criterion: module for loss computation
        self._build_criterion()
        self.criterion = self.criterion.to(rank)

        # build model: bart encoder/decoder & criterion
        self.model = BARTWrapper(self.bart.model, self.criterion)
        self.model = self.model.to(rank)

        if torch.cuda.device_count() > 1 and args.multi_gpu:
            if self.rank == 0:
                self.logger.info("- Let's use {} GPUs !".format(torch.cuda.device_count()))
            self.model = DDP(self.model, device_ids=[rank])
        else:
            if self.rank == 0:
                self.logger.info("- Train the model on single GPU :/")

        self._build_optimizer()
        self._build_scheduler()

    def get_dicts(self):
        assert self.src_dict is not None and self.tgt_dict is not None
        return self.src_dict, self.tgt_dict

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.scheduler.step_update(self.get_num_updates())
        return new_lr

    def _build_criterion(self):
        """Build criterion.
        """
        self.criterion = LabelSmoothedCrossEntropyCriterion(self.args.label_smoothing,
                                                            padding_idx=self.src_dict.pad(),
                                                            reduce=self.args.reduce)

    def _load_bart_model(self):
        """Build BART model."""
        self.logger.info("- loading BART model from: {}".format(self.args.bart_path))
        self.bart = BARTModel.from_pretrained(self.args.bart_path,
                                              checkpoint_file=self.args.checkpoint_file,
                                              data_name_or_path=self.args.data_name_or_path)

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad, self.model.parameters(),
            )
        )
        self.optimizer = FairseqAdam(self.args, params)

    def _build_scheduler(self):
        if self.optimizer is None:
            self._build_optimizer()

        self.scheduler = PolynomialDecaySchedule(self.args, self.optimizer)
        self.scheduler.step_update(0)

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm, aggregate_norm_fn=None)

    def save_model(self, epoch=None):
        """Saves session = weights"""
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        if epoch is not None:
            save_path = os.path.join(self.args.save_dir, 'checkpoint{}.pt'.format(epoch))
        else:
            save_path = os.path.join(self.args.save_dir, 'checkpoint.pt')
        
        torch.save(self.bart.state_dict(), save_path)
        self.logger.info("- model saved at: {}".format(save_path))

    def restore_model(self, model_path):
        """Load pre-trained model.
        """
        # self.bart.load_state_dict(torch.load(model_path))
        # self.model = self.bart.model
        self.logger.info("- model restored from: {}".format(model_path))

    def run_epoch(self, train_dataloader, train_sampler, epoch):
        """Performs one complete pass over the train set and evaluate on devlopment set.

        Args:
            train (DataLoader): training set dataloader.
            epoch (int): index of the current epoch.

        """
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        batch_size = self.args.batch_size
        nbatches = len(train_dataloader)
        
        # progbar stuff for logging
        if self.rank == 0:
            prog = Progbar(target=nbatches)

        # could do this inside the loop
        self._set_seed()

        logging_outputs = []
        for i, batch in enumerate(train_dataloader):
            batch = move_to_cuda(batch)

            loss, nll_loss = self.model(batch["target"], **batch['net_input'])
            loss = loss.mean()  # average losses over all GPUs

            self.optimizer.backward(loss)

            if (i + 1) % self.args.update_freq == 0 or (i + 1) == nbatches:
                # clip accumulated gradients (this should be put inside?)
                # self.clip_grad_norm(self.args.clip_norm)
                clip_grad_norm_(self.model.parameters(), self.args.clip_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # emptying the CUDA cache after the first step can reduce the chance of OOM (Not sure if it's useful)
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()

                self.set_num_updates(self.get_num_updates() + 1)

            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': batch['ntokens'],
                'nsentences': batch['target'].size(0),
            }
            logging_outputs += [logging_output]
            del loss

            if self.rank == 0:
                prog.update(i + 1,
                            values=[("token_loss", logging_output['loss'] / logging_output['ntokens'])],
                            exact=[("lr", self.get_lr()), ("num_updates", self.get_num_updates())])

        return logging_outputs

    def train(self, train_dataloader, train_sampler, dev, samples=None):
        """Train the model and evaluate after each epoch."""
        if self.rank == 0:
            self.logger.info('- start training...')

        best_score, nepoch_no_imprv = 0, 0  # for early stopping
        for epoch in range(self.args.max_epoch):
            if self.rank == 0:
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.args.max_epoch))

            train_sampler.set_epoch(epoch)
            logging_outputs = self.run_epoch(train_dataloader, train_sampler, epoch)

            # evaluate the model
            torch.distributed.barrier()

            if self.rank == 0:
                self.logger.info('- evaluate on development set...')
                metrics = self.evaluate(dev)
            
                msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
                self.logger.info(msg)
                score = metrics["acc"]

                # early stopping and saving best parameters
                if score >= best_score:
                    nepoch_no_imprv = 0
                    self.save_model()
                    best_score = score
                    self.logger.info("- new best score! ")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

            torch.distributed.barrier()

    def evaluate(self, dev):
        """Evaluate model on test set.

        Args:
            dev: instance of class DataLoader

        """
        hypos = self.predict(dev)
        hypotheses_batch = [self.bart.decode(x['tokens']) for x in hypos]

        print("- PREDICTION:")
        for i, h in enumerate(hypotheses_batch):
            print('- #{}: {}'.format(i, h))

            if i == 3:
                break

        return {"acc": 1.0}

    def predict(self, data):
        """Generate summary."""
        self.model.eval()

        batch_size = self.args.eval_batch_size
        generator = self.task.build_generator([self.bart.model], self.args)

        # progress_bar = tqdm(train.batch_iter(batch_size))
        nbatches = (len(data) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        preds = []
        for i, sample in enumerate(data.batch_iter(batch_size)):
            sample = move_to_cuda(sample)

            with torch.no_grad():
                prefix_tokens = sample['net_input']['src_tokens'].new_zeros((sample['id'].shape[0], 1))
                prefix_tokens = prefix_tokens.fill_(self.src_dict.bos()).to(self.device)

                translations = self.task.inference_step(
                    generator,
                    [self.bart.model],
                    sample,
                    prefix_tokens=prefix_tokens
                )

            # Process top predictions
            hypos = [x[0] for x in translations]
            hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
            preds.extend(hypos)

            prog.update(i + 1)

        return preds