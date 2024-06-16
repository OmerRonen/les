# ScaLES: Scalable Latent Exploration Score for Pre-Trained Generative Networks

### [Arxiv link]()

![Alt Text](figures/ScaLES.png)

This repository contains the implementation of ScaLES, a latent exploration score for generative networks, by Omer Ronen, Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk, and Bin Yu.

<details><summary><b>Citation</b></summary>

If you use ScaLES or any of the resources in this repo in your work, please use the following citation:

```bibtex

```

</details>

<details open><summary><b>Table of contents</b></summary>

- [Environment setup](#environment)
- [Datasets and models](#datasets)
- [Replication of results](#rep)
    - [Valid generation](#valid)
    - [Bayesian Optimization](#BO)
- [Calculating ScaLES](#scales)
- [License](#license)

</details>

### Environment setup  <a name="environment"></a>

Using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), first clone the current repository:
```bash
git clone https://github.com/OmerRonen/scales.git
```
Then install the dependencies using:
```bash
conda env create --file environment.yml
conda activate scales
```
To use the [log-expected improvement acquisition function](https://arxiv.org/abs/2310.20708), you would have to manually clone and install the [BoTorch](https://github.com/pytorch/botorch) repository:
```bash
git clone https://github.com/pytorch/botorch.git
cd botorch
pip install -e .
```

### Datasets and models  <a name="datasets"></a>

#### Datasets
This repository uses the expressions and SMILES datasets, both can be downloaded from the [repository](https://github.com/mkusner/grammarVAE/) of the [Grammar Variational Autoencoder](https://arxiv.org/abs/1703.01925) paper. Specifically, the `eq2_grammar_dataset.h5` and `250k_rndm_zinc_drugs_clean.smi` files should be downloaded into the `data/grammar` and `data/molecules` directories, respectively.

#### Models
All the models used in our work can be found in the `trained_models` directory. The following command loads a pre-trained VAE for the expressions dataset:

```python
from scales.nets.utils import DATASETS, get_vae
vae, vae_path = get_vae(DATASETS.expressions)
```

### Replication of results  <a name="rep"></a>

For replicating the results on the molecular datasets (SELFIES and SMILES), we recommend using a GPU to avoid long running times.


#### Valid generation <a name="valid"></a>
The results in Table 1 can be replicated using:
```bash
python -m scales.analysis.ood --dataset <DATASET> --output_dir <DIR>
```

where `<DATASET>` should be replaced with `expressions`, `smiles`, or `selfies`, and `<DIR>` with the output directory.

#### Bayesian Optimization <a name="BO"></a>
The Bayesian Optimization results in Section 4 can be replicated using the following CLI (see help for more details):
```bash
scales.analysis.bo 
```

### Calculating ScaLES  <a name="scales"></a>
If you are interested in calculating ScaLES with a given pre-trained generative model, you can use the following code:

```python
from scales.utils.scales import calculate_scales
from scales.nets.utils import DATASETS, get_vae
vae, vae_path = get_vae(DATASETS.expressions)
X = torch.randn((5, vae.latent_dim))
scales = calculate_scales(X = X, model = VAE.decoder)
```

### License <a name="license"></a>

The code is released under the MIT license; see the [LICENSE](LICENSE) file for details.