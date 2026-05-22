import torch
import logging

import numpy as np
import xarray as xr
import pandas as pd
import torch.nn.functional as F

from typing import Optional

logger = logging.getLogger(__name__)


class GCnPRISM(torch.utils.data.Dataset):
    """
    Dataset with GraphCast and PRISM data 
    """
    
    def __init__(
        self,
        forecast_path: str,
        observation_path: str,
        lead_time: int = 1,
        variables: Optional[list] = None,
        target_variable: str = 'ppt',
        normalize: bool = True,
        interpolate_to_obs_resolution: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            forecast_path: Path to forecast data zarr file
            observation_path: Path to observation data zarr file
            lead_time: Lead time in days (1 = 1 day ahead forecast)
            variables: List of forecast variables to use (if None, use all)
            target_variable: Target variable name in observations
            normalize: Whether to normalize data
            interpolate_to_obs_resolution: Whether to interpolate forecast data to observation resolution
        """
        # Load datasets from zarr files
        logger.info(f"Loading forecast data from {forecast_path}")

        if forecast_path.endswith('.zarr'):
            self.forecast_ds = xr.open_zarr(forecast_path)
        elif forecast_path.endswith('.nc'):
            self.forecast_ds = xr.open_dataset(forecast_path)
        else:
            raise ValueError(f"Unsupported file extension: {forecast_path}")
        
        logger.info(f"Loading observation data from {observation_path}")

        if observation_path.endswith('.zarr'):
            self.observation_ds = xr.open_zarr(observation_path)
        elif observation_path.endswith('.nc'):
            self.observation_ds = xr.open_dataset(observation_path)
        else:
            raise ValueError(f"Unsupported file extension: {observation_path}")
        
        self.lead_time = lead_time
        self.target_variable = target_variable
        self.normalize = normalize
        self.interpolate_to_obs_resolution = interpolate_to_obs_resolution
        
        # Select variables
        if variables is None:
            self.variables = list(self.forecast_ds['var'].values)
        else:
            self.variables = list(variables) 

        self.forecast_ds = self.forecast_ds.sel(var=self.variables)
            
        # Create time matching
        self._create_time_matching()
        
        # Only preload data locally
        self._preload_data_local()
        
        # Calculate normalization stats
        if self.normalize:
            self._calculate_normalization_stats()
            
        logger.info(f"Dataset initialized with {len(self.valid_pairs)} valid forecast-observation pairs")
        logger.info(f"Using {len(self.variables)} forecast variables")
        
    def _create_time_matching(self):
        """Create time matching between forecasts and observations."""
        # Get forecast times
        forecast_times = self.forecast_ds.init.values
        forecast_steps = self.forecast_ds.step.values
        
        # Get observation times
        obs_times = self.observation_ds.date.values
        
        # Convert to pandas for easier manipulation
        forecast_df = pd.DataFrame({
            'init': forecast_times,
            'step': forecast_steps[self.lead_time - 1],

            # Valid time is the time of the forecast, minus 12 hours to account for the 1200 UTC initialization time
            'valid_time': forecast_times + forecast_steps[self.lead_time - 1] - pd.Timedelta(12, "h")
        })
        
        obs_df = pd.DataFrame({'date': obs_times})
        
        # Merge on valid time
        merged = pd.merge(forecast_df, obs_df, left_on='valid_time', right_on='date', how='inner')
        
        self.valid_pairs = []
        for _, row in merged.iterrows():
            init_idx = np.where(forecast_times == row['init'])[0][0]
            obs_idx = np.where(obs_times == row['date'])[0][0]
            self.valid_pairs.append((init_idx, obs_idx))
            
        logger.info(f"Found {len(self.valid_pairs)} valid time matches for lead time {self.lead_time}")
        
    def _preload_data_local(self):
        """Preload data locally (for single worker or testing)."""
        logger.info("Preloading data locally ...")
        # Get unique indices
        init_indices = list(set(pair[0] for pair in self.valid_pairs))
        obs_indices = list(set(pair[1] for pair in self.valid_pairs))
        
        # Preload forecast data
        self.forecast_cache = {}
        for init_idx in init_indices:
            try:
                data = self.forecast_ds.data.isel(
                    init=init_idx, 
                    step=self.lead_time-1
                ).values
                self.forecast_cache[init_idx] = data.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load forecast data for init_idx {init_idx}: {e}")
                
        # Preload observation data
        self.observation_cache = {}
        for obs_idx in obs_indices:
            try:
                data = self.observation_ds[self.target_variable].isel(date=obs_idx).values
                self.observation_cache[obs_idx] = data.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load observation data for obs_idx {obs_idx}: {e}")
                
        logger.info(f"Preloaded {len(self.forecast_cache)} forecast samples and {len(self.observation_cache)} observation samples")
        
    def _calculate_normalization_stats(self):
        """Calculate variable-specific normalization statistics"""
        logger.info("Calculating variable-specific normalization statistics ...")
        
        # Initialize statistics dictionaries
        self.forecast_stats = {}
        self.observation_stats = {}

        # Calculate forecast statistics for each variable directly from cache
        logger.info("Calculating forecast statistics from preloaded cache...")
        # forecast_cache: {init_idx: [C, H, W]}
        for var_idx, var in enumerate(self.variables):
            # Collect all data for this variable across all preloaded samples (no extra copy of all variables)
            var_data_list = []
            for init_idx in sorted(self.forecast_cache.keys()):
                var_data_list.append(self.forecast_cache[init_idx][var_idx])  # [H, W]
            var_data = np.stack(var_data_list, axis=0)  # [N, H, W]
            if var == 'tp_surface_lvl0':
                # Log transformation for precipitation
                transformed_data = np.log(var_data + 1)
                self.forecast_stats[var] = {
                    'mean': np.mean(transformed_data),
                    'std': np.std(transformed_data),
                    'transform': 'log'
                }
                logger.info(f"Forecast ({var_idx+1}/{len(self.variables)}) {var} (log transform) - mean: {self.forecast_stats[var]['mean']:.3f}, std: {self.forecast_stats[var]['std']:.3f}")
            else:
                # Standard scaling for other variables
                self.forecast_stats[var] = {
                    'mean': np.mean(var_data),
                    'std': np.std(var_data),
                    'transform': 'standard'
                }
                logger.info(f"Forecast ({var_idx+1}/{len(self.variables)}) {var} (standard) - mean: {self.forecast_stats[var]['mean']:.3f}, std: {self.forecast_stats[var]['std']:.3f}")

        # Calculate observation statistics from preloaded cache
        logger.info("Calculating observation statistics from preloaded cache...")

        # observation_cache: {obs_idx: [H, W]}
        obs_data_list = []
        for obs_idx in sorted(self.observation_cache.keys()):
            obs_data_list.append(self.observation_cache[obs_idx])
        all_obs_data = np.stack(obs_data_list, axis=0)  # [N, H, W]

        if self.target_variable == 'ppt':
            # Log transformation for precipitation
            transformed_data = np.log(all_obs_data + 1)
            self.observation_stats = {
                'mean': np.nanmean(transformed_data),
                'std': np.nanstd(transformed_data),
                'transform': 'log'
            }
            logger.info(f"Observation {self.target_variable} (log transform) - mean: {self.observation_stats['mean']:.3f}, std: {self.observation_stats['std']:.3f}")
        else:
            # Standard scaling for other variables
            self.observation_stats = {
                'mean': np.nanmean(all_obs_data),
                'std': np.nanstd(all_obs_data),
                'transform': 'standard'
            }
            logger.info(f"Observation {self.target_variable} (standard) - mean: {self.observation_stats['mean']:.3f}, std: {self.observation_stats['std']:.3f}")
    
    def _normalize_forecast_data(self, data):
        """Normalize forecast data using variable-specific statistics."""
        if not self.normalize:
            return data
            
        normalized_data = np.zeros_like(data)
        
        for i, var in enumerate(self.variables):
            var_data = data[i]  # Extract data for this variable
            
            if var == 'tp_surface_lvl0':
                # Log transformation for precipitation
                transformed_data = np.log(var_data + 1)
                normalized_data[i] = (transformed_data - self.forecast_stats[var]['mean']) / self.forecast_stats[var]['std']
            else:
                # Standard scaling for other variables
                normalized_data[i] = (var_data - self.forecast_stats[var]['mean']) / self.forecast_stats[var]['std']
                
        return normalized_data
    
    def _normalize_observation_data(self, data):
        """Normalize observation data using variable-specific statistics."""
        if not self.normalize:
            return data
            
        if self.target_variable == 'ppt':
            # Log transformation for precipitation
            transformed_data = np.log(data + 1)
            return (transformed_data - self.observation_stats['mean']) / self.observation_stats['std']
        else:
            # Standard scaling for other variables
            return (data - self.observation_stats['mean']) / self.observation_stats['std']
    
    def _interpolate_forecast_to_observation_resolution(self, forecast_tensor):
        """
        Interpolate forecast data from coarse resolution to observation resolution using PyTorch.
        
        Args:
            forecast_tensor: torch tensor of shape (num_vars, lat_coarse, lon_coarse)
            
        Returns:
            interpolated_tensor: torch tensor of shape (num_vars, lat_fine, lon_fine)
        """
        if not self.interpolate_to_obs_resolution:
            return forecast_tensor
            
        # Get target spatial dimensions from observation dataset
        target_height = len(self.observation_ds.y)
        target_width = len(self.observation_ds.x)
        
        # Add batch dimension for F.interpolate (expects 4D: batch, channels, height, width)
        # forecast_tensor shape: (num_vars, lat_coarse, lon_coarse)
        # We need: (1, num_vars, lat_coarse, lon_coarse)
        if forecast_tensor.dim() == 3:
            forecast_tensor = forecast_tensor.unsqueeze(0)
        
        # Use bilinear interpolation to match observation resolution
        interpolated_tensor = F.interpolate(
            forecast_tensor,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        
        # Remove batch dimension to return to original shape
        # Result shape: (num_vars, lat_fine, lon_fine)
        if interpolated_tensor.shape[0] == 1:
            interpolated_tensor = interpolated_tensor.squeeze(0)
        
        return interpolated_tensor
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        init_idx, obs_idx = self.valid_pairs[idx]
        
        # Only use local cache
        if init_idx not in self.forecast_cache:
            raise KeyError(f"Forecast data for init_idx {init_idx} not found in cache")
        if obs_idx not in self.observation_cache:
            raise KeyError(f"Observation data for obs_idx {obs_idx} not found in cache")
                
        forecast_data = self.forecast_cache[init_idx].copy()
        obs_data = self.observation_cache[obs_idx].copy()

        # Create the land mask
        ocean_mask = np.isnan(obs_data)
        land_mask_tensor = torch.from_numpy(~ocean_mask).unsqueeze(0)

        # Replace nan values to be zero
        obs_data[ocean_mask] = 0
        
        # Normalize data
        forecast_data = self._normalize_forecast_data(forecast_data)
        obs_data = self._normalize_observation_data(obs_data)
        
        # Ensure data is np.float32 before converting to tensors
        forecast_tensor = torch.from_numpy(forecast_data.astype(np.float32))
        obs_tensor = torch.from_numpy(obs_data.astype(np.float32)).unsqueeze(0)
        
        # Interpolate forecast data to observation resolution
        forecast_tensor = self._interpolate_forecast_to_observation_resolution(forecast_tensor)

        # Additional metadata
        init_time_tensor = torch.tensor(pd.to_datetime(self.forecast_ds.init.values[init_idx]).timestamp(), dtype=torch.float32)
        valid_time_tensor = torch.tensor(pd.to_datetime(self.observation_ds.date.values[obs_idx]).timestamp(), dtype=torch.float32)

        return {
            'coarse': forecast_tensor,  # Low-res forecast input
            'target': obs_tensor,       # High-res observation target
            'init_idx': init_idx,
            'obs_idx': obs_idx,
            'target_land_mask': land_mask_tensor,
            'forecast_init_time': init_time_tensor,
            'valid_time': valid_time_tensor,
        }
    
    def get_data_info(self):
        """Get information about the dataset."""
        info = {
            'num_pairs': len(self.valid_pairs),
            'forecast_shape': self.forecast_ds.data.shape,
            'observation_shape': self.observation_ds[self.target_variable].shape,
            'variables': self.variables,
            'lead_time': self.lead_time,
            'target_variable': self.target_variable,
            'normalize': self.normalize,
            'normalization_type': 'variable_specific',
            'interpolate_to_obs_resolution': self.interpolate_to_obs_resolution
        }
        
        # Add spatial resolution info
        info['spatial_resolution'] = {
            'forecast_coarse': f"{len(self.forecast_ds.latitude)}x{len(self.forecast_ds.longitude)}",
            'observation_fine': f"{len(self.observation_ds.y)}x{len(self.observation_ds.x)}",
            'interpolation_enabled': self.interpolate_to_obs_resolution
        }
        
        # Add normalization info
        if hasattr(self, 'forecast_stats'):
            info['forecast_normalization'] = {
                'num_variables': len(self.forecast_stats),
                'precipitation_transform': 'log(x+1)' if 'tp_surface_lvl0' in self.forecast_stats else 'none',
                'other_variables_transform': 'standard'
            }
        
        if hasattr(self, 'observation_stats'):
            info['observation_normalization'] = {
                'transform': 'log(x+1)' if self.target_variable == 'ppt' else 'standard'
            }
        
        info['cache_size'] = len(self.forecast_cache) if hasattr(self, 'forecast_cache') else 0
            
        return info
    
    def __del__(self):
        """Clean up local cache."""
        if hasattr(self, 'forecast_cache'):
            del self.forecast_cache
        if hasattr(self, 'observation_cache'):
            del self.observation_cache 