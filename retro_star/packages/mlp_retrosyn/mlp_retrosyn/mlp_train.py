import os
from collections import defaultdict
from tqdm import tqdm
from .mlp_policies import train_mlp
from pprint import pprint
if __name__ == '__main__':
    import  argparse
    parser = argparse.ArgumentParser(description="train function for retrosynthesis Planner policies")
    parser.add_argument('--template_path',default= 'data/cooked_data/templates.dat',
                        type=str, help='Specify the path of the template.data')
    parser.add_argument('--template_rule_path', default='data/cooked_data/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_dump_folder',default='./model',
                        type=str, help='specify where to save the trained models')
    parser.add_argument('--fp_dim',default=2048, type=int,
                        help="specify the fingerprint feature dimension")
    parser.add_argument('--batch_size', default=1024, type=int,
                        help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="specify the learning rate")
    args =  parser.parse_args()
    template_path = args.template_path
    template_rule_path = args.template_rule_path
    model_dump_folder = args.model_dump_folder
    fp_dim = args.fp_dim
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    lr = args.learning_rate
    print('Loading data...')
    prod_to_rules = defaultdict(set)
    ### read the template data.
    with open(template_path, 'r') as f:
        for l in tqdm(f, desc="reading the mapping from prod to rules"):
            rule, prod = l.strip().split('\t')
            prod_to_rules[prod].add(rule)
    if not os.path.exists(model_dump_folder):
        os.mkdir(model_dump_folder)
    pprint(args)
    train_mlp(prod_to_rules,
              template_rule_path,
              fp_dim=fp_dim,
              batch_size=batch_size,
              lr=lr,
              dropout_rate=dropout_rate,
              saved_model=os.path.join(model_dump_folder, 'saved_rollout_state_1'))
