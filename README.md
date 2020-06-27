# Retrosynthetic Planning with Retro*

Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search

```bibtex
@inproceedings{chen2020retro,
  title={Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search},
  author={Chen, Binghong and Li, Chengtao and Dai, Hanjun and Song, Le},
  booktitle={The 37th International Conference on Machine Learning (ICML 2020)},
  year={2020}
}
```

#### 1. Setup the environment

##### 1) Download the repository
    
    git clone git@github.com:binghong-ml/retro_star.git
    cd retro_star
    
##### 2) Create a conda environment
    
    conda env create -f environment.yml
    conda activate retro_star_env

#### 2. Download the data

##### 1) Download the building block molecules, pretrained models, and (optional) test data 

Download and unzip the files from this [link](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0), 
and put all the folders (```dataset/```, ```one_step_model/``` and ```saved_models/```) under the ```retro_star``` directory.

#### 3. Install Retro* lib

Install the retrosynthetic planning library with the following commands.

    pip install -e retro_star/packages/mlp_retrosyn
    pip install -e retro_star/packages/rdchiral
    pip install -e .

#### 4. Reproduce experiment results

To plan with Retro*, run the following command,

    cd retro_star
    python retro_plan.py --use_value_fn
    
Ignore the ```--use_value_fn``` option to plan without the learned value function.

You can also train your own value function via,

    python train.py
    

#### 5. Example usage

See ```example.py``` for an example usage.

```python
from retro_star.api import RSPlanner

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)

result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
```
