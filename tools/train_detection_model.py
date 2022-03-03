import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader

from simpleAICV.detection.common import DetectionCollater

from tools.scripts import train_detection, validate_detection
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         compute_flops_and_params, build_optimizer,
                         build_scheduler, build_training_mode)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Detection Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    global logger
    logger = get_logger('train', log_dir)

    set_seed(config.seed)

    local_rank = args.local_rank
    # start init process
    if config.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        torch.cuda.set_device(local_rank)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=config.num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True) if config.distributed else None
    collater = DetectionCollater()
    train_loader = DataLoader(config.train_dataset,
                              batch_size=config.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=config.num_workers,
                              collate_fn=collater.next,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)
    val_loader = DataLoader(config.val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            collate_fn=collater.next)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in [
                    'model', 'criterion', 'decoder', 'train_dataset',
                    'val_dataset'
            ]:
                log_info = f'{key}: {value}'
                logger.info(log_info) if (
                    config.distributed
                    and local_rank == 0) or not config.distributed else None

    gpus_type, gpus_num = torch.cuda.get_device_name(
    ), torch.cuda.device_count()
    log_info = f'gpus_type: {gpus_type}, gpus_num: {gpus_num}'
    logger.info(log_info) if (config.distributed and local_rank
                              == 0) or not config.distributed else None

    model = config.model.cuda()
    criterion = config.criterion.cuda()
    decoder = config.decoder.cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if (config.distributed and local_rank
                                  == 0) or not config.distributed else None

    for name, buffer in model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if (config.distributed and local_rank
                                  == 0) or not config.distributed else None

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    model = build_training_mode(config, model, optimizer)

    start_epoch = 1
    # automatically resume model for training if checkpoint model exist
    '''
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_map, loss, lr = checkpoint['best_map'], checkpoint[
            'loss'], checkpoint['lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_map: {best_map:.3f}, total_loss: {loss:.4f}, lr: {lr:.6f}'
        logger.info(log_info) if (config.distributed and local_rank
                                  == 0) or not config.distributed else None
    '''
    # calculate training time
    start_time = time.time()
    best_map = 0.0

    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) if config.distributed else None
        loss = train_detection(train_loader, model, criterion, optimizer,
                               scheduler, epoch, logger, config)
        log_info = f'train: epoch {epoch:0>3d}, total_loss: {loss:.4f}'
        logger.info(log_info) if (config.distributed and local_rank
                                  == 0) or not config.distributed else None

        result_dict = None
        if epoch in config.eval_epoch or epoch == config.epochs:
            if (config.distributed
                    and local_rank == 0) or not config.distributed:
                result_dict = validate_detection(config.val_dataset,
                                                 val_loader, model, decoder,
                                                 config)
                log_info = f'eval: epoch: {epoch:0>3d}'
                if result_dict:
                    for key, value in result_dict.items():
                        log_info += f', {key}: {value}'
                else:
                    log_info += f', no target detected in testset images!'
                logger.info(log_info) if (
                    config.distributed
                    and local_rank == 0) or not config.distributed else None

        if (config.distributed and local_rank == 0) or not config.distributed:
            # save best map model and each epoch checkpoint
            if result_dict and list(result_dict.values())[0] > best_map:
                torch.save(model.module.state_dict(),
                           os.path.join(checkpoint_dir, 'best.pth'))
                best_map = list(result_dict.values())[0]

            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map,
                    'loss': loss,
                    'lr': scheduler.get_lr()[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

    if (config.distributed and local_rank == 0) or not config.distributed:
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(
                    checkpoint_dir,
                    f'{config.network}-epoch{epoch}-mAP{best_map:.3f}.pth'))

    training_time = (time.time() - start_time) / 3600
    flops, params = compute_flops_and_params(config, model)
    log_info = f'train done. model: {config.network}, flops: {flops}, params: {params}, training time: {training_time:.3f} hours, best_map: {best_map:.3f}'
    logger.info(log_info) if (config.distributed and local_rank
                              == 0) or not config.distributed else None


if __name__ == '__main__':
    main()
