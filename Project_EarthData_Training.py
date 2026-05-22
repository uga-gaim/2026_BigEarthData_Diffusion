"""
Training script for diffusion-based downscaling of weather forecasts.

This script trains a diffusion model to downscale low-resolution weather forecasts
to high-resolution observations using the SR3Unet architecture.
"""

import pickle
import logging
from typing import Optional

import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor
)
from networks.SR3Unet import SR3Unet
from datasets.GCnPRISM import GCnPRISM

import pandas as pd
from datetime import timedelta, datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_dict_to_pickle(d: dict, filepath: str):
    """
    Save a dictionary to a pickle file.

    Args:
        d (dict): The dictionary to save.
        filepath (str): The path to the pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(d, f)
    logger.info(f"Saved dictionary to {filepath}")

def load_dict_from_pickle(filepath: str) -> dict:
    """
    Load a dictionary from a pickle file.
    """
    with open(filepath, 'rb') as f:
        logger.info(f"Loaded dictionary from {filepath}")
        return pickle.load(f)

def create_datasets_by_period(
    forecast_path: str,
    observation_path: str,
    lead_time: int,
    val_period: tuple[datetime, datetime],
    test_period: tuple[datetime, datetime],
    variables: Optional[list] = None,
    target_variable: str = 'ppt',
    normalize: bool = True,
    interpolate_to_obs_resolution: bool = True,
) -> tuple:
    # Create full dataset
    full_dataset = GCnPRISM(
        forecast_path=forecast_path,
        observation_path=observation_path,
        lead_time=lead_time,
        variables=variables,
        target_variable=target_variable,
        normalize=normalize,
        interpolate_to_obs_resolution=interpolate_to_obs_resolution,
    )
    
    # Get dataset info
    dataset_info = full_dataset.get_data_info()
    total_samples = len(full_dataset)
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Dataset info: {dataset_info}")

    train_indices, val_indices, test_indices = [], [], []

    for sample_idx, (init_idx, _) in enumerate(full_dataset.valid_pairs):
        init = pd.to_datetime(full_dataset.forecast_ds.init[init_idx].values) - timedelta(hours=12)
        if val_period[0] <= init <= val_period[1]:
            val_indices.append(sample_idx)
        elif test_period[0] <= init <= test_period[1]:
            test_indices.append(sample_idx)
        else:
            train_indices.append(sample_idx)
    
    assert len(train_indices) > 0, f"No training samples found"
    assert len(val_indices) > 0, f"No validation samples found for period {val_period}"
    assert len(test_indices) > 0, f"No test samples found for period {test_period}"
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, full_dataset

def create_datasets_by_split(
    forecast_path: str,
    observation_path: str,
    lead_time: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    variables: Optional[list] = None,
    target_variable: str = 'ppt',
    normalize: bool = True,
    interpolate_to_obs_resolution: bool = True,
) -> tuple:
    """
    Create train, validation, and test datasets.
    
    Args:
        forecast_path: Path to forecast data zarr file
        observation_path: Path to observation data zarr file
        lead_time: Lead time in days
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        variables: List of forecast variables to use
        target_variable: Target variable name
        normalize: Whether to normalize data
        interpolate_to_obs_resolution: Whether to interpolate to observation resolution
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, full_dataset)
    """
    # Create full dataset
    full_dataset = GCnPRISM(
        forecast_path=forecast_path,
        observation_path=observation_path,
        lead_time=lead_time,
        variables=variables,
        target_variable=target_variable,
        normalize=normalize,
        interpolate_to_obs_resolution=interpolate_to_obs_resolution,
    )
    
    # Get dataset info
    dataset_info = full_dataset.get_data_info()
    total_samples = len(full_dataset)
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Dataset info: {dataset_info}")
    
    # Calculate split indices
    train_size = int(train_split * total_samples)
    val_size = int(val_split * total_samples)
    
    # Create splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, full_dataset

def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 8,
    num_workers: int = 1,
    pin_memory: bool = True
) -> tuple:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def create_callbacks(
    patience: int = 10
) -> list:
    """
    Create training callbacks.
    
    Args:
        patience: Number of epochs to wait before early stopping
        
    Returns:
        List of callbacks
    """
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=5,
            mode='min',
            filename='MonitorValLoss_{epoch:05d}-{val_loss:.5f}'
        ),
        
        ModelCheckpoint(
            monitor='epoch',
            save_top_k=5,
            mode='max',
            filename='MonitorEpoch_{epoch:05d}-{val_loss:.5f}'
        ),
        
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=patience,
            min_delta=0.0,
            verbose=True
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    return callbacks
