import os
import random
import numpy as np
from time import gmtime, strftime, localtime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict
import pandas as pd
from pprint import pprint

def preprocess(X,fp_dim):

    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim),useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return arr


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512,
                 dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x


def top_k_acc(preds, gt,k=1):
    # preds = preds.to(torch.device('cpu'))
    probs, idx = torch.topk(preds, k=k)
    idx = idx.cpu().numpy().tolist()
    gt = gt.cpu().numpy().tolist()
    num = preds.size(0)
    correct = 0
    for i in range(num):
        for id in idx[i]:
            if id == gt[i]:
                correct += 1
    return correct, num


def train_val_test(X,y):
    X_train, y_train = [],[]
    X_val, y_val = [],[]
    X_test, y_test = [],[]
    data_number = len(X)
    idx = np.random.permutation(np.arange(0, data_number))
    train_num = int(0.8 * data_number)
    test_num = int(0.1 * data_number)
    for i in idx[0:train_num]:
        X_train.append(X[i])
        y_train.append(y[i])
    for i in idx[train_num:train_num+test_num]:
        X_val.append(X[i])
        y_val.append(y[i])
    for i in idx[train_num+test_num:]:
        X_test.append(X[i])
        y_test.append(y[i])
    return X_train, X_val,X_test,y_train, y_val,y_test

class OnestepDataset(Dataset):
    def __init__(self, X, y, transform=preprocess, fp_dim=2048):
        super(OnestepDataset, self).__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.fp_dim = fp_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_fp = self.transform(self.X[idx],self.fp_dim)
        return X_fp, self.y[idx]

def dataset_iterator(X,y,
                     batch_size=1024,
                     shuffle=True,
                     num_workers=4,
                     fp_dim=2048):
    dataset = OnestepDataset(X,y,fp_dim=fp_dim)
    def collate_fn(batch):
        X, y = zip(*batch)
        X =  np.array(X)
        y = np.array(y)
        return torch.tensor(X,dtype=torch.float32), torch.tensor(y,dtype=torch.int64)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn)

def train_one_epoch(net, train_loader,
                    optimizer,
                    device,
                    loss_fn,
                    it):
    losses = []
    net.train()
    for X_batch, y_batch in tqdm(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss_v = net(X_batch,y_batch)
        loss_v = loss_v.mean()
        # loss_v = loss_fn(y_hat, y_batch)
        loss_v.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss_v.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
    return losses

def eval_one_epoch(net, val_loader,device):
    net.eval()
    eval_top1_correct, eval_top1_num = 0, 0
    eval_top10_correct, eval_top10_num = 0, 0
    eval_top50_correct, eval_top50_num = 0, 0
    loss = 0.0
    for X_batch, y_batch in tqdm(val_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_hat = net(X_batch)
            loss += F.cross_entropy(y_hat,y_batch).item()
            top_1_correct, num1 = top_k_acc(y_hat, y_batch, k=1)
            top_10_correct, num10 = top_k_acc(y_hat, y_batch, k=10)
            top_50_correct, num50 = top_k_acc(y_hat, y_batch, k=50)
            eval_top1_correct += top_1_correct
            eval_top1_num += num1
            eval_top10_correct += top_10_correct
            eval_top10_num += num10
            eval_top50_correct += top_50_correct
            eval_top50_num += num50
    val_1 = eval_top1_correct/eval_top1_num
    val_10 = eval_top10_correct/eval_top10_num
    val_50 = eval_top50_correct/eval_top50_num
    loss = loss / (len(val_loader.dataset))
    return val_1, val_10, val_50, loss

def train(net, data,
          loss_fn = nn.CrossEntropyLoss(),
          lr = 0.001,
          batch_size=16,
          epochs=100,
          fp_dim = 2048,
          wd=0,
          saved_model='../model/saved_states'):

    it = trange(epochs)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = nn.DataParallel(net)
    net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    X_train, y_train, X_val, y_val = data
    train_loader = dataset_iterator(X_train,y_train,batch_size=batch_size,fp_dim=fp_dim)
    val_loader = None
    if X_val is None:
        val_loader = dataset_iterator(X_train,y_train,batch_size=batch_size,shuffle=False,fp_dim=fp_dim)
    else:
        val_loader = dataset_iterator(X_val,y_val,batch_size=batch_size,shuffle=False,fp_dim=fp_dim)

    best = -1
    for e in it:
        # Iterate batches
        train_one_epoch(net,train_loader,optimizer,device,loss_fn,it)
        ## Do validation after one epoch training.
        val_1,val_10, val_50, loss= eval_one_epoch(net,val_loader,device)
        scheduler.step(loss)
        if best < val_1:
            best = val_1
            state = net.state_dict()
            torch.save(state,saved_model)
        print("\nTop 1: {}  ==> Top 10: {} ==> Top 50: {}, validation loss ==> {}".format(val_1, val_10, val_50, loss))


def load_csv(path, prod_to_rules, template_rules):
    X, y = [], []
    df = pd.read_csv(path)
    num = len(df)
    rnx_smiles = list(df['rxn_smiles'])
    retro_templates = list(df['retro_templates'])
    del df
    for i in tqdm(range(num)):
        rxn = rnx_smiles[i]
        rule = retro_templates[i]
        rule = rule.strip().split('.')
        if len(rule) > 1:
            rule = sorted(rule)
        rule = '.'.join(rule)
        product = rxn.strip().split('>')[-1]
        if rule not in prod_to_rules[product]: continue
        X.append(product)
        y.append(template_rules[rule])
    return X, y


def train_mlp(prod_to_rules, template_rule_path, fp_dim=2048,
              batch_size=1024,
              lr=0.001,
              epochs=100,
              weight_decay=0,
              dropout_rate=0.3,
              saved_model='../model/saved_rollout_state'):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='rollout'):
            rule = l.strip()
            template_rules[rule] = i

    rollout = RolloutPolicyNet(n_rules=len(template_rules),fp_dim=fp_dim,dropout_rate=dropout_rate)
    print('mlp model training...')
    train_path = '../data/uspto_all/proc_train_cano_smiles_w_tmpl.csv'
    test_path = '../data/uspto_all/proc_test_cano_smiles_w_tmpl.csv'
    X_train, y_train = load_csv(train_path, prod_to_rules, template_rules)
    X_test, y_test = load_csv(test_path, prod_to_rules, template_rules)
    time_stamp = strftime("%Y-%m-%d_%H:%M:%S",localtime())
    print('Training size:', len(X_train))
    data = (X_train, y_train,X_test,y_test)
    train(rollout, data,
          fp_dim=fp_dim,
          batch_size=batch_size,
          lr=lr,
          epochs=epochs,
          wd=weight_decay,
          saved_model=saved_model + "_{}_".format(fp_dim) + time_stamp +'.ckpt'
          # saved_model = saved_model + "_{}_{}".format(fp_dim,dropout_rate) + '.ckpt'
          # saved_model=saved_model + "_{}".format(fp_dim) + '.ckpt'
          )

def train_all(prod_to_rules, template_rule_path, fp_dim=2048,
              batch_size=1024,
              lr=0.001,
              epochs=100,
              weight_decay=0,
              dropout_rate=0.3,
              saved_model='../model/saved_rollout_state'):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='rollout'):
            rule = l.strip()
            template_rules[rule] = i

    rollout = RolloutPolicyNet(n_rules=len(template_rules),fp_dim=fp_dim,dropout_rate=dropout_rate)
    print('mlp model training...')
    train_path = '../data/uspto_all/proc_all_cano_smiles_w_tmpl.csv'
    X_train, y_train = load_csv(train_path, prod_to_rules, template_rules)
    time_stamp = strftime("%Y-%m-%d_%H:%M:%S",localtime())
    print('Training size:', len(X_train))
    data = (X_train, y_train,None,None)
    train(rollout, data,
          fp_dim=fp_dim,
          batch_size=batch_size,
          lr=lr,
          epochs=epochs,
          wd=weight_decay,
          saved_model=saved_model + "_{}_".format(fp_dim) + time_stamp +'.ckpt'
          # saved_model = saved_model + "_{}_{}".format(fp_dim,dropout_rate) + '.ckpt'
          # saved_model=saved_model + "_{}".format(fp_dim) + '.ckpt'
          )

def load_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    rollout.load_state_dict(torch.load(state_path,map_location='cpu'))
    return rollout, idx2rule

def load_parallel_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    checkpoint = torch.load(state_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule


if __name__ == '__main__':
    import  argparse
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_path',default= '../data/uspto_all/templates.dat',
                        type=str, help='Specify the path of the template.data')
    parser.add_argument('--template_rule_path', default='../data/uspto_all/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_folder',default='../model',
                        type=str, help='specify where to save the trained models')
    parser.add_argument('--with_all', action='store_true', default=False,
                        help='Specify whether to use all templates.' )
    parser.add_argument('--fp_dim',default=2048, type=int,
                        help="specify the fingerprint feature dimension")
    parser.add_argument('--batch_size', default=1536, type=int,
                        help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help="specify the learning rate")
    args =  parser.parse_args()
    template_path = args.template_path
    template_rule_path = args.template_rule_path
    model_folder = args.model_folder
    with_all = args.with_all
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
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    torch._C._set_cudnn_enabled(False)
    print(torch._C._get_cudnn_enabled())
    pprint(args)
    if not with_all:
        train_mlp(prod_to_rules,
              template_rule_path,
              fp_dim=fp_dim,
              batch_size=batch_size,
              lr=lr,
              dropout_rate=dropout_rate,
              saved_model=os.path.join(model_folder, 'saved_rollout_state_1'))
    else:
        train_all(prod_to_rules,
              template_rule_path,
              fp_dim=fp_dim,
              batch_size=batch_size,
              lr=lr,
              dropout_rate=dropout_rate,
              saved_model=os.path.join(model_folder, 'saved_rollout_state_1_with_all'))