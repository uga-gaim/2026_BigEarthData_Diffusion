import os
import logging
import argparse
from utils import DATA_ROOT

from Project_EarthData_Training import (
    create_datasets_by_period,
    create_data_loaders,
    create_callbacks,
)

from utils.pretrained_selection import opinionated_checkpoint_selection

from models.Direct import Direct
from datasets.Residual import ResidualDataset
from models.Diffusion import Diffusion

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
from datetime import datetime

from utils.notifier import notify
from utils.profiler import Profiler

torch.set_float32_matmul_precision('medium')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

profiler = Profiler()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lead_time', type=int, default=1)
    parser.add_argument('--unet_type', type=str, default='AttRes')
    parser.add_argument('--val_period', type=str, default='2022-10-01_2023-09-30')
    parser.add_argument('--test_period', type=str, default='2023-10-01_2024-09-30')
    parser.add_argument('--gen_members', type=int, default=15)
    return parser.parse_args()

def main(
    lead_time: int = 1,
    unet_type: str = 'SR3',
    val_period: tuple[datetime, datetime] = (datetime(2022, 10, 1), datetime(2023, 9, 30)),
    test_period: tuple[datetime, datetime] = (datetime(2023, 10, 1), datetime(2024, 9, 30)),
    gen_members: int = 1,
):
    profiler.checkpoint("Process initialization")

    if unet_type == 'SR3':
        from networks import SR3Unet as UnetChoice
    elif unet_type == 'AttRes':
        from networks import AttResUnet as UnetChoice
    else:
        raise Exception(f'Unknown unet type: {unet_type}')
    
    args = {
        'io': {
            # 'forecast_path': os.path.join(DATA_ROOT, "derived/GC/daily.zarr"),
            # 'observation_path': os.path.join(DATA_ROOT, "derived/PRISM/PRISM_daily_stable_westus_20190101_20241231.zarr"),
            'forecast_path': os.path.join(DATA_ROOT, "derived/GC/daily.nc"),
            'observation_path': os.path.join(DATA_ROOT, "derived/PRISM/PRISM_daily_stable_westus_20190101_20241231.nc"),
        },
    
        'output': {
            'lead_time': lead_time, # Possible values: 1 - 7
            'target_variable': 'ppt',
        },
    
        'model': {
            'base_channels': 128,
        },
    
        'training': {
            'batch_size': 8,
            'val_period': val_period,
            'test_period': test_period,
    
            'unet_lr': 1e-3,
            'unet_max_epochs': 200,
            'unet_patience': 15,
    
            'corrective_baseline': 'Unet', # Possible values: 'Unet', 'GC'
            'diffusion_lr': 1e-4,
            'diffusion_max_epochs': 2000,
            'diffusion_patience': 500,
        },
    }
    
    logger.info(f"Workflow started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Args: {args}")
    
    session_version = f"test_{args['training']['test_period'][0].strftime('%Y%m%d')}_lead_{args['output']['lead_time']}"
    
    #################
    # Load datasets #
    #################
    
    train_dataset, val_dataset, test_dataset, full_dataset = create_datasets_by_period(
        val_period=args['training']['val_period'],
        test_period=args['training']['test_period'],
        forecast_path=args['io']['forecast_path'],
        observation_path=args['io']['observation_path'],
        lead_time=args['output']['lead_time'],
        target_variable=args['output']['target_variable'],
    )
    
    assert full_dataset.normalize, "Forecast data is not normalized"
    
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args['training']['batch_size'],
    )
    
    profiler.checkpoint("Data loading")
    
    ###############
    # Direct Unet #
    ###############
    
    # Create U-Net model    
    direct_unet = UnetChoice(
        in_channels=len(full_dataset.variables),
        cond_channels=None,
        out_channels=1,
        base_c=args['model']['base_channels'],
    )
    
    logger.info(f"Direct U-Net model created with {len(full_dataset.variables)} input channels")
    logger.info(f"Direct U-Net model architecture ({unet_type}) details:")
    logger.info(f"{direct_unet}")
    
    direct_model = Direct(model=direct_unet, lr=args['training']['unet_lr'], optimizer_cls=torch.optim.Adam)
    
    callbacks = create_callbacks(patience=args['training']['unet_patience'])
    logger_tb = TensorBoardLogger(save_dir='lightning_logs', name='DirectUnet', version=session_version)
    
    trainer = pl.Trainer(
        max_epochs=args['training']['unet_max_epochs'],
        callbacks=callbacks,
        logger=logger_tb,
        gradient_clip_val=1.0,
        accelerator="gpu", devices=1,
    )
    if not os.path.exists(f'./lightning_logs/DirectUnet/{session_version}/checkpoints'):
        logger.info(f"Training direct model for {session_version}")
        trainer.fit(direct_model, train_loader, val_loader)
    
    # Select the best checkpoint file from MonitorValLoss with the lowest validation loss
    best_loss = float('inf')
    selected_ckpt = None
    
    for file in os.listdir(f'./lightning_logs/DirectUnet/{session_version}/checkpoints'):
        if file.startswith('MonitorValLoss_epoch='):
            ckpt_loss = float(file.split('=')[-1].rstrip('.ckpt'))
            if ckpt_loss < best_loss:
                best_loss = ckpt_loss
                selected_ckpt = file
    
    if selected_ckpt is None:
        raise ValueError(f"No Direct Unet model checkpoint found for {session_version}")
    
    direct_model_file = f'./lightning_logs/DirectUnet/{session_version}/checkpoints/{selected_ckpt}'
    direct_model = Direct.load_from_checkpoint(direct_model_file, model=direct_unet)
    logger.info(f"Loaded direct model from checkpoint {direct_model_file} with validation loss {best_loss}")
    
    profiler.checkpoint("Direct Unet training")

    ########################
    # Corrective Diffusion #
    ########################
    
    train_dataset_corr = ResidualDataset(
        train_dataset, pre_compute_on_device='cuda',
        corrective_baseline=args['training']['corrective_baseline'], direct_net=direct_unet,
    )
    
    val_dataset_corr = ResidualDataset(
        val_dataset, pre_compute_on_device='cuda',
        corrective_baseline=args['training']['corrective_baseline'], direct_net=direct_unet,
        residual_normalization_stats=train_dataset_corr.residual_normalization_stats,
    )
    
    test_dataset_corr = ResidualDataset(
        test_dataset, pre_compute_on_device='cuda',
        corrective_baseline=args['training']['corrective_baseline'], direct_net=direct_unet,
        residual_normalization_stats=train_dataset_corr.residual_normalization_stats,
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset_corr,
        val_dataset=val_dataset_corr,
        test_dataset=test_dataset_corr,
        batch_size=args['training']['batch_size'],
    )
    
    if args['training']['corrective_baseline'] == 'GC':
        diffusion_in_channels = len(full_dataset.variables)
    elif args['training']['corrective_baseline'] == 'Unet':
        diffusion_in_channels = len(full_dataset.variables) + 1
    else:
        raise ValueError(f"Invalid corrective baseline: {args['training']['corrective_baseline']}")
    
    corrective_unet = UnetChoice(
        in_channels=1,
        cond_channels=diffusion_in_channels,
        out_channels=1,
        base_c=args['model']['base_channels'],
    )
    
    corrective_model = Diffusion(model=corrective_unet, lr=args['training']['diffusion_lr'], optimizer_cls=torch.optim.Adam)
    
    callbacks = create_callbacks(patience=args['training']['diffusion_patience'])
    logger_tb = TensorBoardLogger(save_dir='lightning_logs', name='CorrDiff', version=session_version)
    
    trainer = pl.Trainer(
        max_epochs=args['training']['diffusion_max_epochs'],
        callbacks=callbacks,
        logger=logger_tb,
        gradient_clip_val=1.0,
        accelerator="gpu",
        devices=1,
    )
    
    if not os.path.exists(f'./lightning_logs/CorrDiff/{session_version}/checkpoints'):
        logger.info(f"Training corrective model for {session_version}")
        trainer.fit(corrective_model, train_loader, val_loader)
    
    # Select the last checkpoint file
    ckpt_dir = f'./lightning_logs/CorrDiff/{session_version}/checkpoints'
    selected_ckpt = opinionated_checkpoint_selection(ckpt_dir)
    
    corrective_model_file = f'./lightning_logs/CorrDiff/{session_version}/checkpoints/{selected_ckpt}'
    corrective_model = Diffusion.load_from_checkpoint(corrective_model_file, model=corrective_unet)
    logger.info(f"Loaded diffusion model from checkpoint {corrective_model_file}")
    
    profiler.checkpoint("Corrective Diffusion training")

    #############################
    # Generate Test Predictions #
    #############################
    
    logger.info(f"Generating test predictions for {len(test_dataset_corr)} samples")

    if gen_members > 1:
        logger.info(f"Generating {gen_members} ensemble members per sample")
        all_residual_predictions_list = []
        
        for member in range(gen_members):
            logger.info(f"Generating ensemble member {member + 1}/{gen_members}")
            member_predictions = trainer.predict(corrective_model, test_loader)
            member_residuals = torch.cat([i['residual'] for i in member_predictions], dim=0)
            member_residuals = member_residuals.clone().cpu().numpy()[np.newaxis, ...]  # (1, num_samples, 1, H, W)
            
            all_residual_predictions_list.append(member_residuals)

        all_residual_predictions_members = np.concatenate(all_residual_predictions_list, axis=0)  # (gen_members, num_samples, 1, H, W)
        all_residual_predictions = np.nanmean(all_residual_predictions_members, axis=0)  # (num_samples, 1, H, W)

    else:
        all_predictions = trainer.predict(corrective_model, test_loader)

        all_residual_predictions = torch.cat([i['residual'] for i in all_predictions], dim=0)
        all_residual_predictions = all_residual_predictions.clone().cpu().numpy()

        all_residual_predictions_members = all_residual_predictions.copy()[np.newaxis, ...]  # (1, num_samples, 1, H, W)

    all_forecast_init_times = torch.cat([i['forecast_init_time'] for i in all_predictions], dim=0)
    all_valid_times = torch.cat([i['valid_time'] for i in all_predictions], dim=0)

    all_forecast_init_times = pd.to_datetime(all_forecast_init_times.cpu().numpy(), unit='s')
    all_valid_times = pd.to_datetime(all_valid_times.cpu().numpy(), unit='s')

    # Load data for test indices
    forecast_data = []
    true_residual_data = []
    land_masks = []
    
    for sample in test_dataset_corr:
        forecast_data.append(sample['coarse'])
        true_residual_data.append(sample['target'])
        land_masks.append(sample['target_land_mask'])
    
    # Stack into tensors
    forecast_tensor = torch.stack(forecast_data, dim=0)  # (num_samples, num_vars, height, width)
    true_residual_tensor = torch.stack(true_residual_data, dim=0)  # (num_samples, 1, height, width)
    land_mask_tensor = torch.stack(land_masks, dim=0)  # (num_samples, 1, height, width)

    obs = true_residual_tensor[:, [0]].clone().cpu().numpy()
    unet = forecast_tensor[:, [-1]].clone().cpu().numpy()
    gc = forecast_tensor[:, [0]].clone().cpu().numpy()

    # They are already in numpy format
    cd = all_residual_predictions[:, [0]]
    cd_ens = all_residual_predictions_members[:, :, [0]]
    
    obs[~land_mask_tensor] = np.nan
    unet[~land_mask_tensor] = np.nan
    gc[~land_mask_tensor] = np.nan
    cd[~land_mask_tensor] = np.nan
    cd_ens[:, ~land_mask_tensor] = np.nan
    
    unet = np.exp(unet * full_dataset.observation_stats['std'] + full_dataset.observation_stats['mean']) - 1
    
    gc = np.exp(gc * full_dataset.forecast_stats['tp_surface_lvl0']['std'] + full_dataset.forecast_stats['tp_surface_lvl0']['mean']) - 1
    gc *= 1000
    
    assert train_dataset_corr.residual_normalization_stats is not None, "Residual normalization stats are not set"
    obs = obs * train_dataset_corr.residual_normalization_stats['std'] + train_dataset_corr.residual_normalization_stats['mean']
    cd = cd * train_dataset_corr.residual_normalization_stats['std'] + train_dataset_corr.residual_normalization_stats['mean']
    cd_ens = cd_ens * train_dataset_corr.residual_normalization_stats['std'] + train_dataset_corr.residual_normalization_stats['mean']

    if args['training']['corrective_baseline'] == 'Unet':
        cd += unet
        cd_ens += unet[np.newaxis, ...]
        obs += unet
    elif args['training']['corrective_baseline'] == 'GC':
        cd += gc
        cd_ens += gc[np.newaxis, ...]
        obs += gc
    else:
        raise Exception(f'Unknown corrective_baseline: {args["training"]["corrective_baseline"]}')
    
    obs[obs < 0] = 0
    unet[unet < 0] = 0
    gc[gc < 0] = 0
    cd[cd < 0] = 0
    cd_ens[cd_ens < 0] = 0

    # Convert to logger
    logger.info(f'RMSE for GC: {np.sqrt(np.nanmean((gc - obs) ** 2)):0.3f}')
    logger.info(f'RMSE for Unet: {np.sqrt(np.nanmean((unet - obs) ** 2)):0.3f}')
    logger.info(f'RMSE for CorrDiff: {np.sqrt(np.nanmean((cd - obs) ** 2)):0.3f}')

    output_path = f'./lightning_logs/CorrDiff/{session_version}/predictions.npz'

    if os.path.exists(output_path):
        logger.warning(f"Output file {output_path} already exists. Overwriting...")
    
    np.savez(
        file=output_path,
        obs=obs, unet=unet, gc=gc, cd=cd, cd_ens=cd_ens,
        test_period=args['training']['test_period'],
        lead_time=args['output']['lead_time'],
        target_variable=args['output']['target_variable'],
        direct_model_file=direct_model_file,
        corrective_model_file=corrective_model_file,
        forecast_init_times=all_forecast_init_times,
        valid_times=all_valid_times,
    )

    profiler.checkpoint("Generating test predictions")
    
    logger.info(f"Workflow complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    profiler.print_summary()


if __name__ == '__main__':
    args = parse_args()

    try:
        main(
            lead_time=args.lead_time,
            unet_type=args.unet_type,
            val_period=tuple(datetime.strptime(date, '%Y-%m-%d') for date in args.val_period.split('_')),
            test_period=tuple(datetime.strptime(date, '%Y-%m-%d') for date in args.test_period.split('_')),
            gen_members=args.gen_members,
        )

    except Exception as e:
        notify(f"Project EarthData failed: {e}")
        raise e
