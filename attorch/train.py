import time
import os
from collections import OrderedDict, deque
import datajoint as dj
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

from .optimizers import RAdam
from .utils import logger, set_seed


def copy_state(model):
    """
    Given PyTorch module `model`, makes a copy of the state onto CPU.
    Args:
        model: PyTorch module to copy state dict of

    Returns:
        A copy of state dict with all tensors allocated on the CPU
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        copy_dict[k] = v.cpu() if v.is_cuda else v.clone()

    return copy_dict


class TimeObjectiveTracker():
    def __init__(self):
        self.tracker = np.array([[time.time(), 0.0]])

    def log_objective(self, obj):
        new_track_point = np.array([[time.time(), obj]])
        self.tracker = np.concatenate(
            (self.tracker, new_track_point), axis=0)

    def finalize(self):
        self.tracker[:, 0] -= self.tracker[0, 0]


def early_stopping(model, objective, interval=5, patience=20, start=0, max_iter=1000,
                   maximize=True, tolerance=1e-5, switch_mode=True, restore_best=True,
                   time_obj_tracker=None):
    """
    Early stopping iterator. When it stops, it restores the best previous state of the model.  

    Args:
        model:     model that is being optimized 
        objective: objective function that is used for early stopping. Must be of the form objective(model)
        interval:  interval at which objective is evaluated to consider early stopping
        patience:  number of times the objective is allow to not become better before the iterator terminates
        start:     start value for iteration (used to check against `max_iter`)
        max_iter:  maximum number of iterations before the iterator terminated
        maximize:  whether the objective is maximized of minimized
        tolerance: margin by which the new objective score must improve to be considered as an update in best score
        switch_mode: whether to switch model's train mode into eval prior to objective evaluation. If True (default),
                     the model is switched to eval mode before objective evaluation and restored to its previous mode
                     after the evaluation.
        restore_best: whether to restore the best scoring model state at the end of early stopping
        time_obj_tracker (TimeObjectiveTracker): 
            for tracking training time & stopping objective

    """
    training_status = model.training

    def _objective(mod):
        if switch_mode:
            mod.eval()
        ret = objective(mod)
        if switch_mode:
            mod.train(training_status)
        return ret

    def finalize(model, best_state_dict):
        old_objective = _objective(model)
        if restore_best:
            model.load_state_dict(best_state_dict)
            print(
                'Restoring best model! {:.6f} ---> {:.6f}'.format(
                    old_objective, _objective(model)))
        else:
            print('Final best model! objective {:.6f}'.format(
                _objective(model)))

    epoch = start
    maximize = float(maximize)
    best_objective = current_objective = _objective(model)
    best_state_dict = copy_state(model)
    patience_counter = 0
    while patience_counter < patience and epoch < max_iter:
        for _ in range(interval):
            epoch += 1
            if time_obj_tracker is not None:
                time_obj_tracker.log_objective(current_objective)
            if (~np.isfinite(current_objective)).any():
                print('Objective is not Finite. Stopping training')
                finalize(model, best_state_dict)
                return
            yield epoch, current_objective

        current_objective = _objective(model)

        if current_objective * (-1) ** maximize < best_objective * (-1) ** maximize - tolerance:
            print('[{:03d}|{:02d}/{:02d}] ---> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
            best_state_dict = copy_state(model)
            best_objective = current_objective
            patience_counter = 0
        else:
            patience_counter += 1
            print('[{:03d}|{:02d}/{:02d}] -/-> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
    finalize(model, best_state_dict)


# ------------ Dynamic training schedule with checkpoints ------------------


def init_save_dict(model, optimizer, scheduler, val_score, patience=10):
    save_dict = {'period': [0], 'epoch': [0], 'lr': [optimizer.param_groups[0]['lr']],
                 'train_score': [None], 'val_score': [val_score], 'val_score_sma': [val_score],
                 'val_score_deque': deque([val_score for _ in range(patience)]),
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'best_period': 0,  'best_epoch': 0,
                 'best_score': val_score, 'num_periods': 0, 'num_lrs': 0}
    return save_dict


def schedule(model, train_func, val_func, seed=0, lr=0.01, mode='min', factor=0.1,
             patience=10, threshold=0.0001, threshold_mode='rel', max_lrs=1, max_epochs=100,
             save_dir='checkpoint', save_dict=None):
    logger.info('\tSeed: {}'.format(seed))
    logger.info('\tLearning rate: {}'.format(lr))
    logger.info('\tMode: {}'.format(mode))
    logger.info('\tFactor: {}'.format(factor))
    logger.info('\tPatience: {}'.format(patience))
    logger.info('\tThreshold: {}'.format(threshold))
    logger.info('\tThreshold mode: {}'.format(threshold_mode))
    logger.info('\tMax lrs: {}'.format(max_lrs))
    logger.info('\tMax epochs: {}'.format(max_epochs))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'final.pt')
    best_path = os.path.join(save_dir, 'best.pt')
    log_path = os.path.join(save_dir, 'log.txt')

    score_sign = {'min': 1, 'max': -1}[mode]
    training_status = model.training
    model.train(True)

    optimizer = RAdam(model.params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                  threshold=threshold, threshold_mode=threshold_mode)

    if save_dict is None:
        # Initialize
        logger.info('Beginning of training')
        model.train(False)
        set_seed(seed)
        with torch.no_grad():
            val_score, val_finite = val_func(epoch_seed=seed)
        save_dict = init_save_dict(model, optimizer, scheduler, val_score, patience)
        from_checkpoint = False
    else:
        # Load
        logger.info('Loading model, optimizer, scheduler from checkpoint')
        model.load_state_dict(save_dict['model'])
        optimizer.load_state_dict(save_dict['optimizer'])
        scheduler.load_state_dict(save_dict['scheduler'])
        val_score = save_dict['val_score'][-1]
        val_finite = True
        from_checkpoint = True

    def step(scheduler, score):
        lr_old = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(metrics=score)
        lr_new = scheduler.optimizer.param_groups[0]['lr']
        save_dict['num_lrs'] += int(not np.allclose(lr_old, lr_new))

    if from_checkpoint:
        step(scheduler, val_score)

    if (scheduler.last_epoch + 1 >= max_epochs) or (save_dict['num_lrs'] >= max_lrs):
        logger.info('Restarting training')
        optimizer = RAdam(model.params, lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                      threshold=threshold, threshold_mode=threshold_mode)
        save_dict['num_periods'] += 1
        save_dict['num_lrs'] = 0

    if not val_finite:
        model.train(training_status)
        return save_dict, False

    torch.backends.cudnn.benchmark = True
    model_finite = True
    while scheduler.last_epoch + 1 < max_epochs:

        period = save_dict['num_periods'] + 1
        epoch = save_dict['epoch'][-1] + 1
        lr = optimizer.param_groups[0]['lr']
        epoch_seed = epoch + seed

        model.train(True)
        set_seed(epoch_seed)
        train_score, train_finite = train_func(optimizer, epoch_seed=epoch_seed, desc=log_path)

        model.train(False)
        set_seed(epoch_seed)
        with torch.no_grad():
            val_score, val_finite = val_func(epoch_seed=epoch_seed)

        if (not train_finite) or (not val_finite):
            model_finite = False
            break

        save_dict['period'].append(period)
        save_dict['epoch'].append(epoch)
        save_dict['lr'].append(lr)
        save_dict['train_score'].append(train_score)
        save_dict['val_score'].append(val_score)
        save_dict['val_score_deque'].popleft()
        save_dict['val_score_deque'].append(val_score)
        save_dict['val_score_sma'].append(np.mean(save_dict['val_score_deque']).item())
        save_dict['model'] = model.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        save_dict['scheduler'] = scheduler.state_dict()

        if save_dict['val_score_sma'][-1] * score_sign < save_dict['best_score'] * score_sign:
            save_dict['best_score'] = save_dict['val_score_sma'][-1]
            save_dict['best_period'] = period
            save_dict['best_epoch'] = epoch
            torch.save(model.state_dict(), best_path, pickle_protocol=3)
        torch.save(save_dict, save_path, pickle_protocol=3)

        log_str = '{} : Current period: {}, Current epoch: {}, Current score: {:.4f}, ' \
            'Best period: {}, Best epoch: {}, Best score: {:.4f}\n'.format(
                time.ctime(), period, epoch, save_dict['val_score_sma'][-1],
                save_dict['best_period'], save_dict['best_epoch'], save_dict['best_score'])
        with open(log_path, 'a') as f:
            f.write(log_str)

        step(scheduler, val_score)

        if save_dict['num_lrs'] >= max_lrs:
            break

        if epoch % 10 == 0:
            assert dj.conn().is_connected

    model.train(training_status)
    return save_dict, model_finite


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0
    if norm_type == math.inf:
        for p in parameters:
            param_norm = p.grad.data.abs().max().item()
            if np.isfinite(param_norm).item():
                total_norm = param_norm if param_norm > total_norm else total_norm
            else:
                return
    else:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type).item()
            if np.isfinite(param_norm).item():
                total_norm += param_norm ** norm_type
            else:
                return
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
