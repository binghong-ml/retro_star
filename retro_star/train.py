import os
import numpy as np
import torch
import random
import pickle
import torch.nn.functional as F
import logging
from retro_star.common import args
from retro_star.model import ValueMLP
from retro_star.data_loader import ValueDataLoader
from retro_star.trainer import Trainer
from retro_star.utils import setup_logger

def train():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    model = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device
    )

    assert os.path.exists('%s/%s.pt' % (args.value_root, args.value_train))

    train_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_train),
        batch_size=args.batch_size
    )

    val_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_val),
        batch_size=args.batch_size
    )

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder,
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('train.log')

    train()
