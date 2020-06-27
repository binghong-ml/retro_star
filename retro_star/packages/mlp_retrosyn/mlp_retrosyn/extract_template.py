"""
Modified version of:
<https://github.com/connorcoley/ochem_predict_nn/blob/master/data/generate_reaction_templates.py>
"""

import os
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
import pandas as pd

if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--data_folder', default='../data/uspto_all/',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--file_name',default='proc_all_cano_smiles_w_tmpl.csv',
                        type=str,
                        help='Specify the filen name')
    args = parser.parse_args()
    data_folder = args.data_folder
    file_name = args.file_name


    templates = defaultdict(tuple)
    transforms = []
    datafile = os.path.join(data_folder,file_name)
    df = pd.read_csv(datafile)
    rxn_smiles = list(df['rxn_smiles'])
    retro_templates = list(df['retro_templates'])
    for i in tqdm(range(len(df))):
        rxn = rxn_smiles[i]
        rule = retro_templates[i]
        product = rxn.strip().split('>')[-1]
        transforms.append((rule,product))
    print(len(transforms))
    with open(os.path.join(data_folder,'templates.dat'), 'w') as f:
        f.write('\n'.join(['\t'.join(rxn_prod) for rxn_prod in transforms]))

    # Generate rules for MCTS
    templates = defaultdict(int)
    for rule, _ in tqdm(transforms):
        templates[rule] += 1
    print("The number of templates is {}".format(len(templates)))
    # #
    template_rules = [rule for rule, cnt in templates.items() if cnt >= 1]
    print("all template rules with count >= 1: ", len(template_rules))
    with open(os.path.join(data_folder,'template_rules_1.dat'), 'w') as f:
        f.write('\n'.join(template_rules))