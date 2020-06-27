import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging


class Trainer:
    def __init__(self, model, train_data_loader, val_data_loader, n_epochs, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _pass(self, data, train=True):
        self.optim.zero_grad()

        for i in range(len(data)):
            data[i] = data[i].to(self.device)

        fps, values, r_costs, t_values, r_fps, r_masks = data
        v_pred = self.model(fps)
        loss = F.mse_loss(v_pred, values)

        batch_size, n_reactants, fp_dim = r_fps.shape
        r_values = self.model(r_fps.view(-1, fp_dim)).view((batch_size,
                                                            n_reactants))
        r_values = r_values * r_masks
        r_values = torch.sum(r_values, dim=1, keepdim=True)

        """
        r_values:   sum of reactant values in a negative reaction sample
        r_costs:    reaction cost
        t_values:   true product value
        7. (const): margin, -log(1e-3)
        """

        r_gap = - r_values - r_costs + t_values + 7.
        r_gap = torch.clamp(r_gap, min=0)
        loss += (r_gap**2).mean()

        if train:
            loss.backward()
            self.optim.step()

        return loss.item()

    def _train_epoch(self):
        self.model.train()

        losses = []
        pbar = tqdm(self.train_data_loader)
        for data in pbar:
            loss = self._pass(data)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()

    def _val_epoch(self):
        self.model.eval()

        losses = []
        pbar = tqdm(self.val_data_loader)
        for data in pbar:
            loss = self._pass(data, train=False)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()

    def train(self):
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            self.train_data_loader.reshuffle()

            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f] [validation loss: %f]' %
                (epoch, self.n_epochs, train_loss, val_loss)
            )

            # if val_loss < best_val_loss or epoch==self.n_epochs-1:
            #     best_val_loss = val_loss
            #     save_file = self.model_folder + '/best_epoch_%d.pt' % epoch
            #     torch.save(self.model.state_dict(), save_file)

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)
