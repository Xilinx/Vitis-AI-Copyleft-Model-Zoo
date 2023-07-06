# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp

import numpy as np
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import tools.eval as model_eval
from yolov6.data.data_load import create_dataloader
from yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6_nndct.models.yolo import ModelNNDctWrapper                                          
from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from torch_rewriters import FUNCTION_REWRITER
from torch_rewriters.utils import IR, Backend


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.__init__', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def Trainer____init__(ctx, self, args, cfg, device):
    self.args = args
    self.cfg = cfg
    self.device = device

    if args.resume:
        self.ckpt = torch.load(args.resume, map_location='cpu')

    self.rank = args.rank
    self.local_rank = args.local_rank
    self.world_size = args.world_size
    self.main_process = self.rank in [-1, 0]
    self.save_dir = args.save_dir
    # get data loader
    self.data_dict = load_yaml(args.data_path)
    self.num_classes = self.data_dict['nc']
    self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
    # get model and optimizer
    model = self.get_model(args, cfg, self.num_classes, device)
    if self.args.distill:
        self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)
        # we use half precision for QAT teacher model caused by not use amp
        if self.args.quant:
            self.teacher_model = self.teacher_model.half()
    if self.args.quant:
        model = self.quant_setup(model, cfg, device)
    if cfg.training_mode == 'repopt':
        scales = self.load_scale_from_pretrained_models(cfg, device)
        reinit = False if cfg.model.pretrained is not None else True
        self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
    else:
        self.optimizer = self.get_optimizer(args, cfg, model)
    self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
    self.ema = ModelEMA(model) if self.main_process else None
    # tensorboard
    self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
    self.start_epoch = 0
    #resume
    if hasattr(self, "ckpt"):
        if self.args.quant:
            resume_state_dict = self.ckpt['nndct_qat_trainable_model']
            model.load_state_dict(resume_state_dict)
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                resume_ema_state_dict = self.ckpt['nndct_qat_trainable_ema_model']
                self.ema.ema.load_state_dict(resume_ema_state_dict)
                self.ema.updates = self.ckpt['updates']
        else:
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            load_state_dict(args.resume, model, map_location=next(model.parameters()).device, state_dict=resume_state_dict)
            # model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                load_state_dict(args.resume, self.ema.ema, map_location=next(self.ema.ema.parameters()).device, state_dict=self.ckpt['ema'].float().state_dict())
                # self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
    self.model = self.parallel_model(args, model, device)
    self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

    self.max_epoch = args.epochs
    self.max_stepnum = len(self.train_loader)
    self.batch_size = args.batch_size
    self.img_size = args.img_size
    self.vis_imgs_list = []
    self.write_trainbatch_tb = args.write_trainbatch_tb
    # set color for classnames
    self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]


    self.loss_num = 3
    self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss']
    if self.args.distill:
        self.loss_num += 1
        self.loss_info += ['cwd_loss']

@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.train_in_steps', backend=Backend.NNDCT.value, ir=IR.XMODEL)
# Training loop for batchdata
def Trainer__train_in_steps(ctx, self, epoch_num):
    images, targets = self.prepro_data(self.batch_data, self.device)
    # plot train_batch and save to tensorboard once an epoch
    if self.write_trainbatch_tb and self.main_process and self.step == 0:
        self.plot_train_batch(images, targets)
        write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')

    # forward
    # for QAT, not to use amp
    with amp.autocast(enabled=self.device != 'cpu' and not self.args.quant):
        preds, s_featmaps = self.model(images)
        if self.args.distill:
            with torch.no_grad():
                if self.args.quant:
                    images = images.half()
                t_preds, t_featmaps = self.teacher_model(images)
                if self.args.quant:
                    t_preds = ([a.float() for a in t_preds[0]], *[a.float() for a in t_preds[1:]])
                    t_featmaps = [a.float() for a in t_featmaps]
            temperature = self.args.temperature
            total_loss, loss_items = self.compute_loss_distill(preds, t_preds, s_featmaps, t_featmaps, targets, \
                                                                epoch_num, self.max_epoch, temperature)
        else:
            total_loss, loss_items = self.compute_loss(preds, targets, epoch_num)
        if self.rank != -1:
            total_loss *= self.world_size
    # backward
    self.scaler.scale(total_loss).backward()
    self.loss_items = loss_items
    self.update_optimizer()


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.quant_setup', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def Trainer__quant_setup(ctx, self, model, cfg, device):
    if self.args.quant:
        from pytorch_nndct import QatProcessor
        inputs = torch.randn([1, 3, self.args.img_size, self.args.img_size],
                       dtype=torch.float32).to(device)
        qat_processor = QatProcessor(
            model, inputs, bitwidth=8, device=device)
        ctx.cfg['qat_processor'] = qat_processor
        ctx.cfg['model'] = model
        quantized_model = qat_processor.trainable_model()

        # if self.main_process:
            # print(quantized_model)
        return quantized_model


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.eval_model', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def Trainer__eval_model(ctx, self):
    model = self.ema.ema if self.args.calib is False else self.model
    if self.args.quant:
        deployable_model = ctx.cfg['qat_processor'].to_deployable(de_parallel(model),
                                                    osp.join(self.save_dir, 'qat'))
        model = ModelNNDctWrapper(None, deployable_model)

    if not hasattr(self.cfg, "eval_params"):
        results, vis_outputs, vis_paths = model_eval.run(self.data_dict,
                        batch_size=self.batch_size // self.world_size * 2,
                        img_size=self.img_size,
                        model=model,
                        conf_thres=0.03,
                        dataloader=self.val_loader,
                        save_dir=self.save_dir,
                        task='train')
    else:
        def get_cfg_value(cfg_dict, value_str, default_value):
            if value_str in cfg_dict:
                if isinstance(cfg_dict[value_str], list):
                    return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                else:
                    return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
            else:
                return default_value
        eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
        results, vis_outputs, vis_paths = model_eval.run(self.data_dict,
                        batch_size=get_cfg_value(self.cfg.eval_params, "batch_size", self.batch_size // self.world_size * 2),
                        img_size=eval_img_size,
                        model=model,
                        conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                        dataloader=self.val_loader,
                        save_dir=self.save_dir,
                        task='train',
                        test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                        letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int", False),
                        force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                        not_infer_on_rect=get_cfg_value(self.cfg.eval_params, "not_infer_on_rect", False),
                        scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                        verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                        do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                        do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric", False),
                        plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                        plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix", False),
                        )

    LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
    self.evaluate_results = results[:2]
    # plot validation predictions
    self.plot_val_pred(vis_outputs, vis_paths)


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.get_data_loader', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def Trainer__get_data_loader(ctx, self, args, cfg, data_dict):
    train_path, val_path = data_dict['train'], data_dict['val']
    # check data
    nc = int(data_dict['nc'])
    class_names = data_dict['names']
    assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
    grid_size = max(int(max(cfg.model.head.strides)), 32)
    # create train dataloader
    train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                        hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                        workers=args.workers, shuffle=True, check_images=args.check_images,
                                        check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
    # create val dataloader
    val_loader = None
    if args.rank in [-1, 0]:
        if args.quant:
            not_infer_on_rect = True
            rect = not not_infer_on_rect
            eval_hyp = {
                "test_load_size": 634,
                "letterbox_return_int": True,
            }
            batch_size = 1
        else:
            rect = True
            eval_hyp = dict(cfg.data_aug)
            batch_size = args.batch_size // args.world_size * 2
        val_loader = create_dataloader(val_path, args.img_size, batch_size, grid_size,
                                        hyp=eval_hyp, rect=rect, rank=-1, pad=0.5,
                                        workers=args.workers, check_images=args.check_images,
                                        check_labels=args.check_labels, data_dict=data_dict, task='val')[0]

    return train_loader, val_loader


@FUNCTION_REWRITER.register_rewriter(func_name='yolov6.core.engine.Trainer.eval_and_save', backend=Backend.NNDCT.value, ir=IR.XMODEL)
def Trainer__eval_and_save(ctx, self):
    remaining_epochs = self.max_epoch - self.epoch
    eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 1
    is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
    if self.main_process:
        self.ema.update_attr(self.model, include=['nc', 'names', 'stride']) # update attributes for ema model
        if is_val_epoch:
            self.eval_model()
            self.ap = self.evaluate_results[1]
            self.best_ap = max(self.ap, self.best_ap)
        # save ckpt
        if self.args.quant:
            model = ctx.cfg['model']
            ckpt = {
                    'model': deepcopy(de_parallel(model)).half(),
                    'nndct_qat_trainable_model': de_parallel(self.model).state_dict(),
                    'nndct_qat_trainable_ema_model': self.ema.ema.state_dict(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    }
        else:
            ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    }

        save_ckpt_dir = osp.join(self.save_dir, 'weights')
        save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
        if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
            save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')

        #default save best ap ckpt in stop strong aug epochs
        if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
            if self.best_stop_strong_aug_ap < self.ap:
                self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')
            
        del ckpt
        # log for learning rate
        lr = [x['lr'] for x in self.optimizer.param_groups] 
        self.evaluate_results = list(self.evaluate_results) + lr
        
        # log for tensorboard
        write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)
        # save validation predictions to tensorboard
        write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')
