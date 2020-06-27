from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import rdchiral
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from .mlp_policies import load_parallel_model , preprocess
from collections import defaultdict, OrderedDict

def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret,key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(templates)



class MLPModel(object):
    def __init__(self,state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path,template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if self.device >= 0:
            arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds,dim=1)
        if self.device >= 0:
            preds = preds.cpu()
        probs, idx = torch.topk(preds,k=topk)
        # probs = F.softmax(probs,dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i , rule in enumerate(rule_k):
            out1 = []
            try:
                out1 = rdchiralRunText(rule, x)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0: continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    templates.append(rule)
            # out1 = rdchiralRunText(x, rule)
            except ValueError:
                pass
        if len(reactants) == 0: return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]
        return {'reactants':reactants,
                'scores' : scores,
                'template' : templates}



if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_rule_path', default='../data/uspto_all/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='../model/saved_rollout_state_1_2048.ckpt',
                        type=str, help='specify where the trained model is')
    args = parser.parse_args()
    state_path = args.model_path
    template_path = args.template_rule_path
    model =  MLPModel(state_path,template_path,device=-1)
    x = '[F-:1]'
    # x = '[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]'
    # x = '[S:3](=[O:4])(=[O:5])([O:6][CH2:7][CH:8]([CH2:9][CH2:10][CH2:11][CH3:12])[CH2:13][CH3:14])[OH:15]'
    # x = 'OCC(=O)OCCCO'
    # x = 'CC(=O)NC1=CC=C(O)C=C1'
    x = 'S=C(Cl)(Cl)'
    # x = "NCCNC(=O)c1ccc(/C=N/Nc2ncnc3c2cnn3-c2ccccc2)cc1"
    # x = 'CCOC(=O)c1cnc2c(F)cc(Br)cc2c1O'
    # x = 'COc1cc2ncnc(Oc3cc(NC(=O)Nc4cc(C(C)(C)C(F)(F)F)on4)ccc3F)c2cc1OC'
    # x = 'COC(=O)c1ccc(CN2C(=O)C3(COc4cc5c(cc43)OCCO5)c3ccccc32)o1'
    x = 'O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3'
    # x = 'CO[C@H](CC(=O)O)C(=O)O'
    # x = 'O=C(O)c1cc(OCC(F)(F)F)c(C2CC2)cn1'
    y = model.run(x,10)
    pprint(y)
