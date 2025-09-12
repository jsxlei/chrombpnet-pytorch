#!/usr/bin/env python

# Author: Lei Xiong <jsxlei@gmail.com>
"""
ChromBPNet Training Script

This script provides functionality for training, predicting, and interpreting ChromBPNet models.

Key features:
- Training models with configurable hyperparameters
- Model prediction and evaluation
- Model interpretation and visualization
- Integration with Weights & Biases for experiment tracking
- Support for distributed training

Usage:
    chrombpnet train [options]  # For training
    chrombpnet predict [options]  # For prediction
    chrombpnet interpret [options]  # For interpretation
"""

import os
import argparse
from typing import Optional, List, Union, Dict, Any

import lightning as L
import torch
import numpy as np
from lightning.pytorch.strategies import DDPStrategy    

# Set precision for matrix multiplication
# torch.set_float32_matmul_precision('medium')

# Set random seed for reproducibility
L.seed_everything(1234)

# Import local modules
from chrombpnet.chrombpnet import BPNet, ChromBPNet
from chrombpnet.model_config import ChromBPNetConfig
from chrombpnet.model_wrappers import create_model_wrapper, load_pretrained_model, adjust_bias_model_logcounts
from chrombpnet.dataset import DataModule
from chrombpnet.data_config import DataConfig
from chrombpnet.genome import hg38_datasets, mm10_datasets
from chrombpnet.metrics import compare_with_observed, save_predictions, load_output_to_regions
from chrombpnet.interpret import run_modisco_and_shap 
from chrombpnet.logger import create_logger

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across train, predict, and interpret commands.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick development test')
    parser.add_argument('--version', '-v', type=str, default=None,
                       help='Version identifier for the run')
    parser.add_argument('--name', type=str, default='',
                       help='Name of the run')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--shap', type=str, default='counts',
                       help='Type of SHAP analysis')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode')
    parser.add_argument('--chrom', type=str, default='test',
                       help='Chromosome to analyze')
    parser.add_argument('--model_type', type=str, default='chrombpnet',
                       help='Type of model to use')
    parser.add_argument('--out_dir', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--alpha', type=float, default=1,
                       help='Alpha value for the model')
    parser.add_argument('--beta', type=float, default=1,
                       help='Beta value for the model')
    parser.add_argument('--bias_scaled', type=str, default=None)
    parser.add_argument('--adjust_bias', action='store_true', default=False,
                       help='Adjust bias model')
    parser.add_argument('--chrombpnet_wo_bias', type=str, default=None,
                       help='ChromBPNet model without bias')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Verbose output')
    parser.add_argument('--precision', type=int, default=32,
                            help='Training precision (16, 32, or 64)')
    parser.add_argument('--max_epochs', type=int, default=100,
                            help='Maximum number of training epochs')
    parser.add_argument('--gradient_clip', type=float, default=None,
                            help='Gradient clipping value')
    parser.add_argument('--force', action='store_true', default=False,
                            help='Force training even if model already exists')
    
    # Add model-specific arguments
    ChromBPNetConfig.add_argparse_args(parser)
    DataConfig.add_argparse_args(parser)

def get_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description='Train or test ChromBPNet model.')
    subparsers = parser.add_subparsers(dest='command')

    # Train sub-command
    train_parser = subparsers.add_parser('train', help='Train the ChromBPNet model.')
    add_common_args(train_parser)

    # Predict sub-command
    predict_parser = subparsers.add_parser('predict', help='Test or predict with the ChromBPNet model.')
    add_common_args(predict_parser)

    # Interpret sub-command
    interpret_parser = subparsers.add_parser('interpret', help='Interpret the ChromBPNet model.')

    finetune_parser = subparsers.add_parser('finetune', help='Finetune the ChromBPNet model.')
    add_common_args(finetune_parser)

    add_common_args(interpret_parser)

    reproduce_parser = subparsers.add_parser('reproduce', help='Reproduce the ChromBPNet model.')
    add_common_args(reproduce_parser)

    predict_bias_parser = subparsers.add_parser('predict_bias', help='Predict bias with the ChromBPNet model.')
    add_common_args(predict_bias_parser)

    add_common_args(parser)

    parser.set_defaults(command='pipeline')

    return parser

def load_model(args):
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}')
    if args.checkpoint is None:
        checkpoint = os.path.join(out_dir, 'checkpoints/best_model.ckpt')
        if not os.path.exists(checkpoint):
            args.checkpoint = None
            print(f'No checkpoint found in {out_dir}/checkpoints/best_model.ckpt')
        else:
            args.checkpoint = checkpoint
            print(f'Loading checkpoint from {checkpoint}')
    model = load_pretrained_model(args)
    return model

def compare_predictions(out_dir, chrom):
    import pandas as pd
    df_tf = pd.read_csv(os.path.join(out_dir, 'reproduce', chrom, 'regions.csv'), sep='\t')
    df_pt = pd.read_csv(os.path.join(out_dir, 'predict', chrom, 'regions.csv'), sep='\t')
    df_tf_peaks = df_tf[df_tf['is_peak']==1]
    df_pt_peaks = df_pt[df_pt['is_peak']==1]
    
    from chrombpnet.metrics import counts_metrics
    # counts_metrics(df_tf['pred_count'], df_pt['pred_count'],outf=os.path.join(out_dir, 'reproduce', chrom, 'compare_counts.png'), title='',
    #     fontsize=20, xlab='Log Count chrombpnet original', ylab='Log Count pytorch')
    counts_metrics(df_tf_peaks['pred_count'], df_pt_peaks['pred_count'],outf=os.path.join(out_dir, 'reproduce', chrom, 'compare'), title='',
        fontsize=20, xlab='Log Count chrombpnet original', ylab='Log Count pytorch')

def train(args):
    data_config = DataConfig.from_argparse_args(args)
    loggers=[L.pytorch.loggers.CSVLogger(args.out_dir, name=args.name, version=f'fold_{args.fold}')]
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, 'checkpoints/best_model.ckpt')) and not args.force:
        raise ValueError(f"Model folder {out_dir}/checkpoints/best_model.ckpt already exists. Please delete the existing model or specify a new version.")
    if args.bias_scaled is None:
        args.bias_scaled = os.path.join(args.data_dir, 'bias_scaled.h5')
    log = create_logger(args.model_type, ch=True, fh=os.path.join(out_dir, 'train.log'), overwrite=True)
    log.info(f'out_dir: {out_dir}')
    log.info(f'bias: {args.bias_scaled}')      
    log.info(f'adjust_bias: {args.adjust_bias}')
    log.info(f'data_type: {data_config.data_type}')
    log.info(f'in_window: {data_config.in_window}') 
    log.info(f'data_dir: {data_config.data_dir}')
    log.info(f'negative_sampling_ratio: {data_config.negative_sampling_ratio}')
    log.info(f'fold: {args.fold}')
    log.info(f'n_filters: {args.n_filters}')
    log.info(f'batch_size: {data_config.batch_size}')
    log.info(f'precision: {args.precision}')

    datamodule = DataModule(data_config)

    args.alpha = datamodule.median_count / 10
    log.info(f'alpha: {args.alpha}')


    model = create_model_wrapper(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())


    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, # 5
        accelerator='gpu',
        devices=args.gpu,
        val_check_interval=None,
        # strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True),
        ],
        logger=loggers, # L.pytorch.loggers.TensorBoardLogger
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        # precision="bf16"
    )
    trainer.fit(model, datamodule)
    if args.model_type == 'chrombpnet' and not args.fast_dev_run:
        torch.save(model.model.model.state_dict(), os.path.join(out_dir, 'checkpoints/chrombpnet_wo_bias.pt'))

def finetune(args):
    data_config = DataConfig.from_argparse_args(args)
    loggers=[L.pytorch.loggers.CSVLogger(args.out_dir, name=args.name, version=f'fold_{args.fold}')]
    out_dir = os.path.join(args.out_dir, args.name, 'finetune', f'fold_{args.fold}')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, 'checkpoints/best_model.ckpt')) and not args.force:
        raise ValueError(f"Model folder {out_dir}/checkpoints/best_model.ckpt already exists. Please delete the existing model or specify a new version.")
    if args.bias_scaled is None:
        args.bias_scaled = os.path.join(args.data_dir, 'bias_scaled.h5')
    log = create_logger(args.model_type, ch=True, fh=os.path.join(out_dir, 'train.log'), overwrite=True)
    log.info(f'out_dir: {out_dir}')
    log.info(f'bias: {args.bias_scaled}')      
    log.info(f'adjust_bias: {args.adjust_bias}')
    log.info(f'data_type: {data_config.data_type}')
    log.info(f'in_window: {data_config.in_window}') 
    log.info(f'data_dir: {data_config.data_dir}')
    log.info(f'negative_sampling_ratio: {data_config.negative_sampling_ratio}')
    log.info(f'fold: {args.fold}')
    log.info(f'n_filters: {args.n_filters}')
    log.info(f'batch_size: {data_config.batch_size}')
    log.info(f'precision: {args.precision}')

    datamodule = DataModule(data_config)

    args.alpha = datamodule.median_count / 10
    log.info(f'alpha: {args.alpha}')


    # model = create_model_wrapper(args)
    model = load_model(args)
    if args.adjust_bias:
        adjust_bias_model_logcounts(model.model.bias, datamodule.negative_dataloader())


    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, # 5
        accelerator='gpu',
        devices=args.gpu,
        val_check_interval=None,
        # strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model', save_last=True),
        ],
        logger=loggers, # L.pytorch.loggers.TensorBoardLogger
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        # precision="bf16"
    )
    trainer.fit(model, datamodule)
    if args.model_type == 'chrombpnet' and not args.fast_dev_run:
        torch.save(model.model.model.state_dict(), os.path.join(out_dir, 'checkpoints/chrombpnet_wo_bias.pt'))


def predict(args, model, datamodule=None, mode='predict'):
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}')
    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        datamodule = DataModule(data_config)

    trainer = L.Trainer(logger=False, fast_dev_run=args.fast_dev_run, devices=args.gpu, val_check_interval=None) 
    os.makedirs(os.path.join(out_dir, mode), exist_ok=True)
    log = create_logger(args.model_type, ch=True, fh=os.path.join(out_dir, f'predict.log'), overwrite=True)
    log.info(f'out_dir: {out_dir}')
    log.info(f'model_type: {args.model_type}')
    log.info(f'checkpoint: {args.checkpoint}')
    log.info(f'peaks: {data_config.peaks}')
    log.info(f'negatives: {data_config.negatives}')
    log.info(f'bigwig: {data_config.bigwig}')
    log.info(f'fasta: {data_config.fasta}')
    log.info(f'chrom_sizes: {data_config.chrom_sizes}')
    log.info(f'chrom: {args.chrom}')

    chrom = args.chrom
    dataloader, dataset = datamodule.chrom_dataloader(args.chrom)
    # log.info(f"{chrom}: {regions['is_peak'].value_counts()}")
    output = trainer.predict(model, dataloader)
    regions, parsed_output = load_output_to_regions(output, dataset.regions, os.path.join(out_dir, mode, chrom))
    if mode == 'predict_bias':
        return
    # import pdb; pdb.set_trace()
    model_metrics = compare_with_observed(regions, parsed_output, os.path.join(out_dir, mode, chrom))     
    save_predictions(output, regions, data_config.chrom_sizes, os.path.join(out_dir, mode, chrom))


def interpret(args, model, datamodule=None):
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}')
    if datamodule is None:
        data_config = DataConfig.from_argparse_args(args)
        datamodule = DataModule(data_config)
    dataloader, dataset = datamodule.chrom_dataloader(args.chrom)
    regions = dataset.regions
    model.to(f'cuda:{args.gpu[0]}')

    tasks = ['profile', 'counts'] if args.shap == 'both' else [args.shap]
    for task in tasks:
        run_modisco_and_shap(model.model.model, data_config.peaks, out_dir=os.path.join(out_dir, 'interpret'), batch_size=args.batch_size,
            in_window=data_config.in_window, out_window=data_config.out_window, task=task, debug=True)
    # out = model._mutagenesis(dataloader, debug=args.debug)
    # os.makedirs(os.path.join(out_dir, 'interpret'), exist_ok=True)
    # np.save(os.path.join(out_dir, 'interpret', 'mutagenesis.npy'), out)

def reproduce(args):
    out_dir = os.path.join(args.out_dir, args.name, f'fold_{args.fold}', 'reproduce')
    os.makedirs(out_dir, exist_ok=True)
    log = create_logger(args.model_type, ch=True, fh=os.path.join(out_dir, f'reproduce.log'), overwrite=True)
    log.info(f'out_dir: {out_dir}')
    log.info(f'model_type: {args.model_type}')
    log.info(f'chrom: {args.chrom}')

    if os.path.exists(os.path.join(args.out_dir, args.name, f'fold_{args.fold}', 'checkpoints/chrombpnet_wo_bias.pt')):
        log.info(f'Model already trained, loading from {args.out_dir}/checkpoints/chrombpnet_wo_bias.pt')
        args.chrombpnet_wo_bias = os.path.join(args.out_dir, args.name, f'fold_{args.fold}', 'checkpoints/chrombpnet_wo_bias.pt')
    else:
        train(args)

    predict_path = os.path.join(args.out_dir, args.name, f'fold_{args.fold}', 'predict', args.chrom, 'regions.csv')
    if not os.path.exists(predict_path):
        log.info(f'Predicting with pytorch model')
        model_wrapper = load_model(args)
        predict(args, model_wrapper)
    else:
        log.info(f'{predict_path} already exists')
        
    if os.path.exists(os.path.join(args.data_dir, f'models/fold_{args.fold}/chrombpnet_wo_bias.h5')):
        log.info(f'Predicting with chrombpnet model: {args.chrombpnet_wo_bias}')
        args.checkpoint = os.path.join(args.data_dir, f'models/fold_{args.fold}/chrombpnet_wo_bias.h5')
        reproduce_path = os.path.join(args.out_dir, args.name, f'fold_{args.fold}', 'reproduce', args.chrom, 'regions.csv')
        if not os.path.exists(reproduce_path):
            model_wrapper = load_model(args)
            predict(args, model_wrapper, mode='reproduce')
        else:
            log.info(f'{reproduce_path} already exists')

        compare_predictions(os.path.join(args.out_dir, args.name, f'fold_{args.fold}'), args.chrom)
    else:
        log.info(f'ChromBPNet model not found in {args.data_dir}/models/fold_{args.fold}/chrombpnet_wo_bias.h5')




def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'train':
        train(args)
        model_wrapper = load_model(args)
        predict(args, model_wrapper)
    elif args.command == 'predict':
        model_wrapper = load_model(args)
        predict(args, model_wrapper)
    elif args.command == 'interpret':
        model_wrapper = load_model(args)
        interpret(args, model_wrapper)
    elif args.command == 'finetune':
        finetune(args)
        model_wrapper = load_model(args)
        predict(args, model_wrapper)
    elif args.command == 'pipeline':
        train(args)
        model_wrapper = load_model(args)
        predict(args, model_wrapper)
        interpret(args, model_wrapper)
    elif args.command == 'reproduce':
        reproduce(args)
    elif args.command == 'predict_bias':
        from chrombpnet.model_wrappers import BPNetWrapper
        # model_wrapper = load_model(args)
        bpnet_bias_wrapper = BPNetWrapper(args)
        bpnet_bias_wrapper.model = bpnet_bias_wrapper.init_bias(args.bias_scaled)
        print('bias_scaled', args.bias_scaled)
        print('bpnet_bias_wrapper', bpnet_bias_wrapper.model)
        predict(args, bpnet_bias_wrapper, mode='predict_bias')

    else:
        raise ValueError(f'Invalid command: {args.command}')


                

if __name__ == '__main__':
    main()