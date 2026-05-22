import numpy as np
import torch

from torch.utils.data import Dataset

from typing import Optional
from tqdm.auto import tqdm


class ResidualDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        pre_compute_on_device='cuda',
        corrective_baseline: str = 'GC',
        direct_net: Optional[torch.nn.Module] = None,
        residual_normalization_stats: Optional[dict] = None,
    ):
        assert corrective_baseline in ['GC', 'Unet']
        self.base_dataset = base_dataset
        self.pre_compute_on_device = pre_compute_on_device
        self.corrective_mode = corrective_baseline
        self.direct_net = direct_net
        self.residual_normalization_stats = residual_normalization_stats

        # Check for variables attribute
        variables = None
        if hasattr(self.base_dataset, 'variables'):
            variables = self.base_dataset.variables
        elif hasattr(self.base_dataset, 'dataset') and hasattr(self.base_dataset.dataset, 'variables'):
            variables = self.base_dataset.dataset.variables
        if variables is None:
            raise AttributeError("base_dataset or its underlying dataset must have a 'variables' attribute.")
            
        self.baseline_precip_indx = variables.index('tp_surface_lvl0')

        # Check for normalization in the base dataset
        self.normalized_base_dataset = None
        if hasattr(self.base_dataset, 'normalize'):
            self.normalized_base_dataset = self.base_dataset.normalize
        elif hasattr(self.base_dataset, 'dataset') and hasattr(self.base_dataset.dataset, 'normalize'):
            self.normalized_base_dataset = self.base_dataset.dataset.normalize
        if self.normalized_base_dataset is None:
            raise AttributeError("base_dataset or its underlying dataset must have a 'normalize' attribute.")

        if self.normalized_base_dataset:
            if hasattr(self.base_dataset, 'observation_stats'):
                self.base_dataset_observation_stats = self.base_dataset.observation_stats
                self.base_dataset_forecast_stats = self.base_dataset.forecast_stats
            elif hasattr(self.base_dataset.dataset, 'observation_stats'):
                self.base_dataset_observation_stats = self.base_dataset.dataset.observation_stats
                self.base_dataset_forecast_stats = self.base_dataset.dataset.forecast_stats
            else:
                raise AttributeError("base_dataset or its underlying dataset must have a 'observation_stats' attribute.")

        # Check for __len__
        if not hasattr(self.base_dataset, '__len__'):
            raise AttributeError("base_dataset must implement __len__.")

        self._pre_compute()

    def __len__(self):
        if not hasattr(self.base_dataset, '__len__'):
            raise AttributeError("base_dataset must implement __len__.")
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.pre_calc_samples[idx]

    def _pre_compute(self):

        if self.corrective_mode == 'Unet':
            if self.direct_net is None:
                raise ValueError("direct_net must be provided for Unet corrective mode.")
            if not callable(self.direct_net):
                raise TypeError("direct_net must be callable.")
            self.direct_net.eval()
            self.direct_net.to(self.pre_compute_on_device)

        all_residuals = []
        self.pre_calc_samples = []

        # First pass: collect all residuals in physical space
        for idx in tqdm(range(len(self.base_dataset)), desc='Computing residuals', leave=False):
            batch = self.base_dataset[idx]
            coarse = batch['coarse']        # [C, H, W]
            target = batch['target']        # [1, H, W]
            land_mask = batch['target_land_mask']  # [1, H, W]

            if self.corrective_mode == 'GC':
                benchmark_precip = coarse[[self.baseline_precip_indx]]  # [1, H, W]
            elif self.corrective_mode == 'Unet':
                assert self.direct_net is not None, "direct_net must be provided for Unet corrective mode."
                with torch.no_grad():
                    benchmark_precip = self.direct_net(coarse.unsqueeze(0).to(self.pre_compute_on_device)).cpu().squeeze(0)  # [1, H, W]
            else:
                raise ValueError(f"Invalid corrective mode: {self.corrective_mode}")

            if self.normalized_base_dataset:
                target_mm = target * self.base_dataset_observation_stats['std'] + self.base_dataset_observation_stats['mean']
                target_mm = torch.exp(target_mm) - 1

                if self.corrective_mode == 'GC':
                    benchmark_precip_mm = benchmark_precip * self.base_dataset_forecast_stats['tp_surface_lvl0']['std'] + \
                                          self.base_dataset_forecast_stats['tp_surface_lvl0']['mean']
                    benchmark_precip_mm = torch.exp(benchmark_precip_mm) - 1
                    benchmark_precip_mm *= 1000
                elif self.corrective_mode == 'Unet':
                    benchmark_precip_mm = benchmark_precip * self.base_dataset_observation_stats['std'] + self.base_dataset_observation_stats['mean']
                    benchmark_precip_mm = torch.exp(benchmark_precip_mm) - 1
                else:
                    raise ValueError(f"Invalid corrective mode: {self.corrective_mode}")
                    
            else:
                target_mm = target

                if self.corrective_mode == 'GC':
                    benchmark_precip_mm = benchmark_precip * 1000
                elif self.corrective_mode == 'Unet':
                    benchmark_precip_mm = benchmark_precip

            residual = target_mm - benchmark_precip_mm

            # Ignore residuals over ocean by assigning them to np.nan
            residual[~land_mask] = np.nan
            all_residuals.append(residual.numpy())

            # if corrective mode is Unet, add the direct prediction as an extra channel to the batch
            if self.corrective_mode == 'Unet':
                batch['coarse'] = torch.cat([coarse, benchmark_precip], dim=0)

            self.pre_calc_samples.append(batch)

        # Compute mean and std over all residuals
        all_residuals_np = np.concatenate(all_residuals)  # [N, H, W]

        if self.normalized_base_dataset:
            if self.residual_normalization_stats is None:
                mean_r = np.nanmean(all_residuals_np)
                std_r = np.nanstd(all_residuals_np)

                self.residual_normalization_stats = {
                    'mean': mean_r,
                    'std': std_r,
                    'transform': 'standard'
                }
            else:
                mean_r = self.residual_normalization_stats['mean']
                std_r = self.residual_normalization_stats['std']

            # Normalize residuals
            all_residuals_np = (all_residuals_np - mean_r) / std_r

        # Substitute nan values with 0
        all_residuals_np[np.isnan(all_residuals_np)] = 0

        # Second pass: simply replace the target with the normalized residuals in the pre_calc_samples
        for idx in tqdm(range(len(self.pre_calc_samples)), desc='Replacing targets with normalized residuals', leave=False):
            self.pre_calc_samples[idx]['target'] = torch.from_numpy(all_residuals_np[idx]).unsqueeze(0)

        if self.corrective_mode == 'Unet':
            self.direct_net.to('cpu')
            torch.cuda.empty_cache()

    def get_data_info(self):
        """Get information about the dataset."""
        info = {
            'num_pairs': len(self.pre_calc_samples),
            'corrective_mode': self.corrective_mode,
            'residual_normalization_stats': self.residual_normalization_stats,
        }
        return info