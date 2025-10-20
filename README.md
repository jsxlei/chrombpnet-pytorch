[![Stars](https://img.shields.io/github/stars/jsxlei/chrombpnet-pytorch?logo=GitHub&color=yellow)](https://github.com/jsxlei/chrombpnet-pytorch/stargazers)
[![PyPI](https://img.shields.io/pypi/v/chrombpnet-pytorch.svg)](https://pypi.org/project/chrombpnet-pytorch)
<!-- [![Documentation Status](https://readthedocs.org/projects/chrombpnet-pytorch/badge/?version=latest)](https://chrombpnet-pytorch.readthedocs.io/en/latest/?badge=stable) -->
[![Downloads](https://pepy.tech/badge/chrombpnet-pytorch)](https://pepy.tech/project/chrombpnet-pytorch)
# ChromBPNet Pytorch

- Pytorch implementation for [ChromBPNet](https://github.com/kundajelab/chrombpnet)
- Please refer to original [code](https://github.com/kundajelab/chrombpnet) and paper [ChromBPNet: Bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v1) by  Anusri Pampari*, Anna Shcherbina*, Anshul Kundaje. (*authors contributed equally)  
- This repo also refers to [bpnet-lite](https://github.com/jmschrei/bpnet-lite) and uses [tangermeme](https://github.com/jmschrei/tangermeme) for interpretation, two very useful repos by Jacob Schreiber


<p align="center">
<img src="images/chrombpnet_arch.png" alt="ChromBPNet" align="center" style="width: 800px;"/>
</p>

## Reproduce Official ChromBPNet performance
### Pearson correlation on counts prediction of peaks
official chrombpnet (left) vs pytorch chrombpnet (right)
<p align="center">
  <img src="images/peaks.counts_pearsonr_official.png" width="350" />
  <img src="images/peaks.counts_pearsonr.png" width="350" /> 
</p>

### Attribution score
Here is the [genome browser](https://epigenomegateway.wustl.edu/browser2022/?blob=7Vhdc5s4FP0rGe2rDcaJvY7fupm0zXSbnYk92YedDiPQBVQLiQrhxPXkv_cKMMYOTt1u291p84bQuV_nXF0.1iQGqVK4pimQKUni0wnpkSWHuytpQC.pINM1yQ3VhkzH3vn52XjUIyBZtRqPTh96xGgaLnIy_WdNZOXnphCg0ZFZZXap66WgAaDDZltlhiuJluu9LXRaaLtEUAqGMmqoRc1tpJMdrwjl.QwEhAYwq4iKHHok4gL.Ct5XHuzCpveulSpe83wO91hWbWJwcaFkxGOMhF5pxtvLDwXo1aVkmeISjdYPD72mXA3RK5CwLRhJBSqlMtQW2Kp8i.yofbN5VPV7If4zGnp1B226p0ULboSKwe3p.RHM7IA7yGnt_yz8vH1xfennZU6.5wyOIMlanFQWuHGyrKw62OoC_iy03UAG1Lyl.WJnxpR3083dZp7sgrsmThvRcJQYk.VT113yj0kROHdFboQDrHCzIhA8dG1Srk7zhTd2guDJMdVO7P8wrbacBTy.Q4vWIdSqkMw3ujDJgWO4i1jNQiqsK1oYZddv6T2ZesPBIyJTjtU4.CSRkdKspHKBnuh7cBVduAL4PcaKN_xWpbsaYgnGfTMaD53gbo_m3Q7p7dJe1_b1hJcUNiEQ9687Ggm51FrpWgMb6wVjrbSqAwtsToMryQCZHPSIrRZbsOzUVBk4KQ3bByLTwHwVRTzkVPh0aRU9pHAXtkPmLtiu1t9IYKus2w7k29DPWj.pdZholQYZngw_GnnDz8n9GH5I8cfI7yd6tjJKh4n_EgM9i_5lontDS8fRolfwY0SvkD9oqtvovHxBycvr_UyeG.JQQzQDs8WYPVR.iE9n4.cJzbb9wVYyhw.t_jjOuqNdjjP8LiMDK8Ao0mgeOGW03ImUYH6Kb6vO5fXF7GYynrx89cZpTsPRfVPz80sMkvbMPb5hPmPWNViesviGTxF_J1DVGGWQL5wdv2IP4IT9mh7oNnuqB7osnnvgx_cAprghYw747VpljQ9jJHYGZn99W_4FLJ32vTr6n4D1sb85w68_.0LQIwGKJOAKUyKT0WjIwrNBf3A6CvueFw36k_Fg3A.j4dnZ7zCJzkP7cwfZTdQdl_E1XfKYonZkit.TWE2zc3vTVCjoShWWDhILFdT_IzPBDf6enPGPyDQmYWiACb8GykC_Bh4naNDcRm5m.NzImg0kdut185WubK9wW8ZvHl6FCRdMgyz_atYYdJeDaWD29fgAbIOhmT0b9cH7A2Pk5V.IUKWZkmBVriEgaSDgQqgckWXlVi2KL0dLvGHZeXiHaQd4NECXQuEqwXqEralWjlG9mCdgg1UuPgE-) to compare the profile prediction and attribution scores between official ChromBPNet and pytorch implementation with n_filters = 512 and 128
<p align='center'>
<img src="images/genome_brower.png" alt="genome_browser", align="center" style="width: 1200px;"/>
</p>


## Table of contents

- [Installation](#installation)
- [QuickStart](#quickstart)
- [How-to-cite](#how-to-cite)

## Installation

#### Install from pypi 

```
pip install chrombpnet-pytorch
```
#### Install from source
```
pip install git+https://github.com/jsxlei/chrombpnet-pytorch.git
```

## QuickStart
### Before training
- Download the [genome](https://zenodo.org/records/12193595) or use your own genome data
- Download the [ENCODE K562 ATAC data](https://zenodo.org/records/15713376) or use your own ATAC data

### Bias-factorized ChromBPNet training
Please refer to [data_config](https://github.com/kundajelab/chrombpnet-pytorch/chrombpnet/data_config.py) to define your own dataset or pass them through command.

if your <data_path> contains: peaks.bed, negatives.bed, and unstranded.bw, and bias_scaled.h5 as well.
```
chrombpnet train --data_dir <data_path>
```
Otherwise
```
chrombpnet train --peaks <peak_file> --negatives <negative_file> --bigwig <unstrand.bw> --bias <bias_scaled.h5> --adjust_bias
```

### Predict with pretrained model in .h5 format or .cpkt or .pt
```
chrombpnet predict --data_dir <data_path> --checkpoint chrombpnet_wo_bias.h5/best_model.cpkt/chrombpnet_wo_bias.pt -o <output_path>
```

### Interpret by calculating attribution
```
chrombpnet interpret --data_dir <data_path> --checkpoint <model_cpkt/model_h5> -o <output_path>
```

### Run full pipeline including training, predicting and interpreting
```
chrombpnet --data_dir <data_path> -o <output_path>
```

### Finetune model
```
chrombpnet finetune --data_dir <data_path> --checkpoint <model>.h5/pt -o <output_path>
```

### Variant scoring
```
snp_score 
 	-l $snps \
 	-g $ref_fasta \
 	-pg $ref_fasta_peaks \
 	-s $chrom_sizes \
 	-ps $chrom_sizes_peaks \
 	-m $model \
 	-p $peaks \
 	-o $out_prefix \
 	-t 2 \
 	-li \
 	-sc chrombpnet
```
#### Input Format

- `--bigwig` 
- `--peaks`
- `--negatives`


#### Output Format

The ouput directory will be populated as follows with fold_0 chromosome splits -

```
fold_0\
	checkpoints\
		best_model.cpkt
		last.cpkt
		chrombpnet_nobias.pt (pytorch i.e model to predict bias corrected accessibility profile) 
	train.log
	predict.log
	evaluation\
		eval\
			all_regions.counts_pearsonr.png
			all_regions_jsd.profile_jsd.png  
			peaks.counts_pearsonr.png  
			peaks_jsd.profile_jsd.png  
			regions.csv
			metrics.json
	interpret\
		counts\



```

 
## How to Cite

If you're using ChromBPNet in your work, please cite as follows:

```
@article {Pampari2024.12.25.630221,
	author = {Pampari, Anusri and Shcherbina, Anna and Kvon, Evgeny and Kosicki, Michael and Nair, Surag and Kundu, Soumya and Kathiria, Arwa S. and Risca, Viviana I. and Kuningas, Kristiina and Alasoo, Kaur and Greenleaf, William James and Pennacchio, Len A. and Kundaje, Anshul},
	title = {ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants},
	elocation-id = {2024.12.25.630221},
	year = {2024},
	doi = {10.1101/2024.12.25.630221},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/12/25/2024.12.25.630221},
	eprint = {https://www.biorxiv.org/content/early/2024/12/25/2024.12.25.630221.full.pdf},
	journal = {bioRxiv}
}
```