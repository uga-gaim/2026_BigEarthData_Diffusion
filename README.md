# Comparing Analog Ensemble and Corrective Diffusion for Western US Precipitation Downscaling from GraphCast

This repository contains the source code for the paper:

> Hu, W., Yang, Z., Baño-Medina, J., Sengupta, A., Afzali Gorooh, V., & Delle Monache, L. (2026). Comparing Analog Ensemble and Corrective Diffusion for Western US precipitation downscaling from GraphCast. *Big Earth Data*, 1–26. https://doi.org/10.1080/20964471.2026.2665935

## Citation

```bibtex
@article{hu2026comparing,
  title={Comparing Analog Ensemble and Corrective Diffusion for Western US precipitation downscaling from GraphCast},
  author={Hu, Weiming and Yang, Zhiqi and Ba{\~n}o-Medina, Jorge and Sengupta, Agniv and Afzali Gorooh, Vesta and Delle Monache, Luca},
  journal={Big Earth Data},
  pages={1--26},
  year={2026},
  publisher={Taylor \& Francis}
}
```

## Repository Structure

```
.
├── Project_EarthData_main.py       # Main training entry point
├── Project_EarthData_Training.py   # Training utilities
├── Project_EarthData.ipynb         # Experiment notebook
├── Project_EarthData_main.job      # HPC job script (SLURM)
├── Project_EarthData_main.sh       # Shell runner
├── Project_EarthData_main.submit   # HPC submit script
├── AIWP_Download.ipynb             # Download AI weather prediction data
├── GraphCast_Restructure.ipynb     # Restructure GraphCast output
├── PRISM_Download.ipynb            # Download PRISM precipitation data
├── PRISM_Restructure.ipynb         # Restructure PRISM data
├── datasets/                       # Dataset classes
├── models/                         # Diffusion and Direct model definitions
└── utils/                          # Utility functions
```

## Environment Setup

```bash
conda create -n venv_torch python==3.11 -y
conda activate venv_torch

conda install -c conda-forge jupyterlab ipywidgets tqdm matplotlib scipy ipykernel seaborn jupyterlab-lsp python-lsp-server -y
pip3 install torch torchvision torchaudio "xarray[complete]" cloudpickle cfgrib rioxarray rasterio zarr "geopandas[all]" "lightning[extra]" pynvml tensorboard s3fs ipython discord_notify yaspin
```

## Data

| Dataset | Description | Source |
|---|---|---|
| GraphCast | Global AI weather prediction output | [Google DeepMind](https://github.com/google-deepmind/graphcast) |
| PRISM | High resolution precipitation analysis for the Western US | [PRISM Climate Group](https://prism.oregonstate.edu) |

## Other References

[Analog Ensemble](https://weiming.uga.edu/AnalogEnsemble/)