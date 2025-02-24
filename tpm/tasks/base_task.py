"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from tpm.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from tpm.common.logger import MetricLogger, SmoothedValue
from tpm.common.registry import registry
from tpm.datasets.data_utils import prepare_sample

import torch.nn as nn

import numpy as np

def xyz_values_to_category(x, y, z):
    value_to_index = {
        -0.01: 0,
        -0.005: 1,
        0: 2,
        0.005: 3,
        0.01: 4
    }
    x_index = value_to_index[x]
    y_index = value_to_index[y]
    z_index = value_to_index[z]

    category = x_index + 5 * y_index + 25 * z_index
    return category


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):

        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]


            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            #dataset['valid'].name = name

            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):

        img0 = samples['image_view_0']
        img1 = samples['image_view_1']

        #mask0 = samples['mask_view_0']
        mask0 = samples['mask_view_0'][:, 1:2, :, :]
        mask0 = torch.cat([mask0, img0], dim=1)

        #mask1 = samples['mask_view_1']
        mask1 = samples['mask_view_1'][:, 1:2, :, :]
        mask1 = torch.cat([mask1, img1], dim=1)

        position1 = samples['position1']
        position2 = samples['position2']
        state = samples['state']

        joint_value = samples['joint_value']
        output = model(image0=img0, mask0=mask0, image1=img1, mask1=mask1, pose=state, joint_values=joint_value)


        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        mse_loss = nn.MSELoss(reduction='none')

        action_gt = samples['action_xyz']
        griper_gt = samples['action_griper'].view(-1).type(torch.long)
        weight = samples['weights']




        loss_action = mse_loss(output[0], action_gt)
       
        loss_action = loss_action.mean(dim=0)
        loss_action = loss_action.mean(dim=0)

       
        loss_griper = cross_entropy_loss(output[1], griper_gt)
        loss_griper = loss_griper.mean(dim=0)



     
        loss = 5000*loss_action + 5 * loss_griper


        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            #print("######################################################################")
            #print(use_amp)
            use_amp = False
            if use_amp:
                scaler.scale(loss).backward()
            else:
                if not torch.isnan(loss):
                    # 如果损失不是NaN，则进行反向传播和权重更新
                    loss.backward()
                    metric_logger.update(loss=loss.item())
                else:
                    # 如果损失是NaN，跳过此批次的反向传播和权重更新，并输出警告信息
                    print("Warning: Loss is NaN. Skipping weight update for this batch.")
                #loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            #metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file


class BaseTask_multi:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):

        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        print("构建数据集")

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]


            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):

        img0 = samples['image_view_0']
        img1 = samples['image_view_1']

        mask0 = samples['mask_view_0'][:,:,1:2,:,:]
        mask0 = torch.cat([mask0, img0], dim=2)

        mask1 = samples['mask_view_1'][:,:, 1:2, :, :]
        mask1 = torch.cat([mask1, img1], dim=2)

        position1 = samples['position1']
        position2 = samples['position2']
        state = samples['state']

        #output = model(image0=img0, mask0=mask0, image1=img1, mask1=mask1)
        output = model(image0=img0, mask0=mask0, image1=img1, mask1=mask1, pose=state)

        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        mse_loss = nn.MSELoss(reduction='none')

        action_gt = samples['action_xyz']
        griper_gt = samples['action_griper'].view(-1).type(torch.long)
        weight = samples['weights']
        # weight = 1.0

        # cross_entropy_loss2 = nn.CrossEntropyLoss(reduction='none')
        # action_gt2 = samples['action_xyz']
        # value_to_index = torch.tensor([-0.01, -0.005, 0, 0.005, 0.01])
        # value_to_index = value_to_index.to(action_gt.device)
        # x_index = (action_gt2[:, 0].view(-1, 1) == value_to_index).nonzero()[:, 1]
        # y_index = (action_gt2[:, 1].view(-1, 1) == value_to_index).nonzero()[:, 1]
        # z_index = (action_gt2[:, 2].view(-1, 1) == value_to_index).nonzero()[:, 1]
        # # 计算类别索引
        # action_gt_categories = x_index + 5 * y_index + 25 * z_index
        # # 计算交叉熵损失
        # loss_action2 = cross_entropy_loss(output[2], action_gt_categories)
        # loss_action2 = loss_action2.mean(dim=0)
        # loss_action2 = loss_action2.mean(dim=0)

        # loss_pos1 = mse_loss(output[2], position1[:,:3])*weight.unsqueeze(1)
        # loss_pos2 = mse_loss(output[3], position2[:,:3])*weight.unsqueeze(1)
        # loss_pos1 = loss_pos1.mean(dim=0)
        # loss_pos1 = loss_pos1.mean(dim=0)
        # loss_pos2 = loss_pos2.mean(dim=0)
        # loss_pos2 = loss_pos2.mean(dim=0)

        loss_action = mse_loss(output[0], action_gt)
        # loss_action = mse_loss(output[0], action_gt)*weight.unsqueeze(1)
        loss_action = loss_action.mean(dim=0)
        loss_action = loss_action.mean(dim=0)

        # loss_griper = cross_entropy_loss(output[1], griper_gt)*weight
        loss_griper = cross_entropy_loss(output[1], griper_gt)
        loss_griper = loss_griper.mean(dim=0)

        # binary_gt = torch.zeros_like(action_gt)
        # binary_gt[action_gt > 0] = 1
        # sigmoid = nn.Sigmoid()
        # output_prob = sigmoid(output[0])
        # bce_loss = nn.BCELoss()
        # loss_direction = bce_loss(output_prob, binary_gt)

        # loss = loss_action + loss_griper + loss_pos1 + loss_pos2

        loss = 5000 * loss_action + 5 * loss_griper  # + loss_action2 #+ 20*loss_pos1 + 20*loss_pos2

        # loss = loss_pos2 + loss_pos1

        # loss = 1000*loss_pos1 + 500*loss_pos2

        # loss = 5000 * loss_action

        # print(loss_action, loss_griper, loss_pos1, loss_pos2)

        print(loss_action, loss_griper)

        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            #print("######################################################################")
            #print(use_amp)
            use_amp = False
            if use_amp:
                scaler.scale(loss).backward()
            else:
                if not torch.isnan(loss):
                    # 如果损失不是NaN，则进行反向传播和权重更新
                    loss.backward()
                else:
                    # 如果损失是NaN，跳过此批次的反向传播和权重更新，并输出警告信息
                    print("Warning: Loss is NaN. Skipping weight update for this batch.")
                #loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file






