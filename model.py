from math import dist
from typing import Any, Optional

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl

# import pytorch_lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder
import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn.pytorch.glob import Set2Set
from audtorch.metrics.functional import pearsonr


def similarity_score(x, y):
    distance = torch.norm(x - y, dim=-1)
    score = 1 / (1 + distance)
    return score


def pearson_score(a, b):
    a_mu = a - a.mean(dim=-1).unsqueeze(-1)
    b_mu = b - b.mean(dim=-1).unsqueeze(-1)
    return (a_mu * b_mu).clamp(-0x7FFF, 0x7FFF).mean(dim=-1) / (
        a.std(correction=0, dim=-1) * b.std(correction=0, dim=-1)
    )


class MyModel(nn.Module):
    def __init__(
        self,
        in_feature: int,
        hidden_feature: int,
        out_feature: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        adj_len: int,
    ):
        super().__init__()

        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.adj_len = adj_len

        self.conv1 = GATv2Conv(
            in_feats=self.in_feature,
            out_feats=self.hidden_feature,
            num_heads=self.num_heads,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
            negative_slope=self.alpha,
        )
        self.conv2 = GATv2Conv(
            in_feats=self.hidden_feature * self.num_heads,
            out_feats=self.hidden_feature,
            num_heads=1,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
            negative_slope=self.alpha,
        )
        self.nlp = nn.Linear(self.hidden_feature * self.adj_len, 128)
        # self.read_out = Set2Set(self.hidden_feature, n_iters=3, n_layers=3)

    def forward(self, g):
        h = g.ndata["feat"]
        batch_size = h.shape[0] // self.adj_len
        h = self.conv1(g, h)
        nodes, heads, output_features = h.shape
        h = torch.reshape(h, (nodes, heads * output_features))
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.squeeze(-2)
        h = F.elu(h)
        h = self.nlp(h.view(batch_size, -1))
        return h


class PLModelForAST(pl.LightningModule):
    def __init__(
        self,
        adj_length: int,
        pool_size: int = 0,
        lr: float = 5e-5,
        in_features=64,
        hidden_features=128,
        output_features=64,
        n_heads=4,
        dropout=0.6,
        alpha=0.2,
        seed=1,
        data_path="",
    ):
        super().__init__()
        self.lr = lr
        self.pool_size = pool_size
        self.adj_length = adj_length
        # def __init__(self, config, seed: int = 3407, pool_size: int = 50) -> None:
        #     super().__init__()
        #     self.lr = config['lr']
        #     in_features = config['in_features']
        #     hidden_features = config['hidden_features']
        #     output_features = config['output_features']
        #     n_heads = config['n_heads']
        #     self.adj_length = config['adj_length']
        #     alpha = config['alpha']
        #     dropout = config['dropout']
        #     self.pool_size = pool_size
        self.seed = seed
        self.my_model = MyModel(
            in_feature=in_features,
            hidden_feature=hidden_features,
            out_feature=output_features,
            num_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            adj_len=self.adj_length,
        )
        # self.my_model = torch.compile(self.my_model)
        self.validation_acc_outputs = np.array([])
        self.validation_diff_outputs = np.array([])
        self.validation_loss_all_outputs = np.array([])
        self.validation_loss_1v1_outputs = np.array([])
        self.validation_loss_pool_outputs = np.array([])

        self.training_acc_outputs = np.array([])
        self.training_diff_outputs = np.array([])
        # self.data_path = data_path
        self.save_hyperparameters()

    def forward(self, x):
        torch.cuda.empty_cache()
        if self.pool_size:
            sample, same, diff, label, pool = (
                x["sample"],
                x["same_sample"],
                x["different_sample"],
                x["label"],
                x["pool"],
            )
        else:
            sample, same, diff, label = (
                x["sample"],
                x["same_sample"],
                x["different_sample"],
                x["label"],
            )

        seq = [1, 2, 3]
        random.shuffle(seq)
        for s in seq:
            if s == 1:
                latent_sample = self.my_model(sample)
            elif s == 2:
                latent_same = self.my_model(same)
            else:
                latent_diff = self.my_model(diff)

        # latent size = [batch, output_size]

        # loss1 = torch.abs(F.cosine_similarity(latent_sample, latent_same, dim=-1) - 1).mean()
        # loss1 = 1 - similarity_score(latent_sample, latent_same).mean()
        loss1 = (1 - abs(pearson_score(latent_sample, latent_same))).mean()

        # loss2 = similarity_score(latent_sample, latent_diff).mean()

        # loss2 = F.cosine_embedding_loss(latent_sample, latent_diff, label - 1)
        loss2 = abs(pearson_score(latent_sample, latent_diff)).mean()

        with torch.no_grad():
            # cosine_same = F.cosine_similarity(latent_same, latent_sample, dim=-1).detach().cpu().numpy() # [batch]
            # cosine_diff = F.cosine_similarity(latent_diff, latent_sample, dim=-1).detach().cpu().numpy() # [batch]
            # same = similarity_score(latent_same, latent_sample).detach().cpu().numpy()
            same = abs(pearson_score(latent_same, latent_sample)).detach().cpu().numpy()

            # different = similarity_score(latent_diff, latent_sample).detach().cpu().numpy()
            different = (
                abs(pearson_score(latent_diff, latent_sample)).detach().cpu().numpy()
            )

            # Detect NAN output
            if (
                np.isnan(same).any()
                or np.isnan(different).any()
                or np.isinf(same).any()
                or np.isinf(different).any()
            ):
                diff = None
                is_right = None
            else:
                diff = same - different
                is_right = (diff > 0).astype(int)

        if self.pool_size:
            batch_size, output_size = latent_same.shape[0], latent_same.shape[1]
            pool_latents = []
            for b in range(batch_size):
                pool_latent = self.my_model(pool[b])
                pool_latents.append(pool_latent)
            pool_latents = torch.vstack(pool_latents)
            pool_latents = pool_latents.view(
                batch_size, -1, output_size
            )  # [batch_size, pool_size, output_size]
            pool_latents = torch.concat([pool_latents, latent_same.unsqueeze(1)], dim=1)
            # similarity = F.cosine_similarity(latent_sample.unsqueeze(1), pool_latents, dim=-1)
            # similarity = similarity_score(latent_sample.unsqueeze(1), pool_latents)
            similarity = abs(pearson_score(latent_sample.unsqueeze(1), pool_latents))

            loss3 = F.cross_entropy(
                similarity,
                torch.tensor([self.pool_size] * batch_size, dtype=torch.long).to(
                    device=self.device
                ),
            )

            return (
                loss1 + loss2 + loss3,
                (loss1 + loss2).item(),
                loss3.item(),
                is_right,
                diff,
            )

        return (loss1 + loss2, 0.0, is_right, diff)

    def training_step(self, batch, batch_idx):

        if batch_idx == 347:
            print("Debug")

        if self.pool_size:

            loss_all, loss_1v1, loss_pool, ok, diff = self.forward(batch)
            if torch.isnan(loss_all).any():
                return None
            self.log(
                "train_loss_1v1",
                loss_1v1,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "train_loss_pool",
                loss_pool,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=True,
            )

        else:
            loss_all, ok, diff = self.forward(batch)
            if torch.isnan(loss_all).any():
                return None

        self.log(
            "train_loss_all",
            loss_all.item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train_diff",
            np.mean(diff),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.training_acc_outputs = np.append(self.training_acc_outputs, ok)
        self.training_diff_outputs = np.append(self.training_diff_outputs, diff)
        return loss_all

    def validation_step(self, batch, batch_idx):
        if self.pool_size:
            loss_all, loss_1v1, loss_pool, ok, diff = self.forward(batch)
            self.validation_loss_1v1_outputs = np.append(
                self.validation_loss_1v1_outputs, loss_1v1
            )
            self.validation_loss_pool_outputs = np.append(
                self.validation_loss_pool_outputs, loss_pool
            )
            self.validation_loss_all_outputs = np.append(
                self.validation_loss_all_outputs, loss_all.item()
            )

        else:
            loss_all, ok, diff = self.forward(batch)

        self.validation_acc_outputs = np.append(self.validation_acc_outputs, ok)
        self.validation_diff_outputs = np.append(self.validation_diff_outputs, diff)

        return loss_all

    def on_validation_epoch_end(self):
        acc = np.mean(self.validation_acc_outputs)
        self.log(
            "val_acc",
            acc.item(),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )

        diff = np.mean(self.validation_diff_outputs)
        self.log(
            "val_diff",
            diff.item(),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )

        self.validation_acc_outputs = np.array([])
        self.validation_diff_outputs = np.array([])

        if self.pool_size:
            loss_1v1 = np.mean(self.validation_loss_1v1_outputs)
            self.log(
                "val_loss_1v1",
                loss_1v1.item(),
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.validation_loss_1v1_outputs = np.array([])

            loss_all = np.mean(self.validation_loss_all_outputs)
            self.log(
                "val_loss_all",
                loss_all.item(),
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.validation_loss_all_outputs = np.array([])

            loss_pool = np.mean(self.validation_loss_pool_outputs)
            self.log(
                "val_loss_pool",
                loss_pool.item(),
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.validation_loss_pool_outputs = np.array([])

    def on_train_epoch_end(self):
        acc = np.mean(self.training_acc_outputs)
        self.log(
            "train_acc",
            acc.item(),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )

        diff = np.mean(self.training_diff_outputs)
        self.log(
            "train_diff",
            diff.item(),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )

        self.training_acc_outputs = np.array([])
        self.training_diff_outputs = np.array([])

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
