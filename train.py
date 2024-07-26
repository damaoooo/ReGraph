import torch
import os
import yaml
import pickle
from dataclasses import dataclass
import argparse
from lightning.pytorch import Trainer, seed_everything
from model import PLModelForAST
from dataset import ASTGraphDataModule, ASTGraphRedisDataModule
from lightning.pytorch.callbacks import ModelCheckpoint

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"
# torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class TrainConfig:
    batch_size: int = 1
    num_workers: int = 8
    pool_size: int = 5
    data_path: str = "/opt/li_dataset/binutils"
    lr: float = 4e-4
    alpha: float = 0.2
    dropout: float = 0.3
    hidden_features: int = 64
    n_heads: int = 6
    output_features: int = 128
    load_checkpoint: str = None
    seed: int = 1
    max_epochs: int = 200
    k_fold: int = 0
    exclusive_arch: str = None
    exclusive_opt: str = None
    redis: bool = False
    data_name: str = None
    mode: str = "file"


def read_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        f.close()
    return config


def parse_args():
    argparser = argparse.ArgumentParser()
    # configuration file
    argparser.add_argument(
        "--config",
        type=str,
        default="./train_config.yaml",
        help="path to configuration file",
    )
    argparser.add_argument("--batch_size", type=int, default=1, help="batch size")
    argparser.add_argument("--max_epochs", type=int, default=200, help="max epochs")
    argparser.add_argument("--num_workers", type=int, default=8)
    argparser.add_argument("--pool_size", type=int, default=5)
    argparser.add_argument("--k_fold", type=int, default=0)
    argparser.add_argument(
        "--data_path",
        type=str,
        default="/opt/li_dataset/binutils",
        help="path to dataset, should be the folder",
    )
    argparser.add_argument("--lr", type=float, default=4e-4)
    argparser.add_argument("--alpha", type=float, default=0.2)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--hidden_features", type=int, default=64)
    argparser.add_argument("--n_heads", type=int, default=6)
    argparser.add_argument("--output_features", type=int, default=128)
    argparser.add_argument("--load_checkpoint", type=str, default=None)
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument("--exclusive_arch", type=str, default=None)
    argparser.add_argument("--exclusive_opt", type=str, default=None)
    argparser.add_argument("--redis", type=bool, default=False)
    argparser.add_argument("--data_name", type=str, default=None)
    argparser.add_argument("--mode", type=str, default="file")
    return argparser.parse_args()


def read_config() -> TrainConfig:
    config: TrainConfig = TrainConfig()
    args = parse_args()
    if args.config is not None:
        yaml_config = read_yaml_config(args.config)
        # TODO: Read yaml file and update config
        print("Reading Configuration From YAML ......")
        config.alpha = yaml_config["hyper_parameters"]["alpha"]
        config.lr = yaml_config["hyper_parameters"]["lr"]
        config.dropout = yaml_config["hyper_parameters"]["dropout"]
        config.hidden_features = yaml_config["hyper_parameters"]["hidden_features"]
        config.n_heads = yaml_config["hyper_parameters"]["n_heads"]
        config.output_features = yaml_config["hyper_parameters"]["output_features"]

        config.data_path = yaml_config["path"]["data_path"]
        config.load_checkpoint = yaml_config["path"]["load_checkpoint"]
        config.data_name = yaml_config["path"]["data_name"]

        config.batch_size = yaml_config["train"]["batch_size"]
        config.num_workers = yaml_config["train"]["num_workers"]
        config.pool_size = yaml_config["train"]["pool_size"]
        config.seed = yaml_config["train"]["seed"]
        config.max_epochs = yaml_config["train"]["max_epochs"]
        config.k_fold = yaml_config["train"]["k_fold"]

        config.exclusive_arch = yaml_config["hyper_parameters"]["exclusive_arch"]
        config.exclusive_opt = yaml_config["hyper_parameters"]["exclusive_opt"]

        config.redis = yaml_config["hyper_parameters"]["redis"]
        config.mode = yaml_config["hyper_parameters"]["mode"]

    else:
        config.alpha = args.alpha
        config.lr = args.lr
        config.dropout = args.dropout
        config.hidden_features = args.hidden_features
        config.n_heads = args.n_heads
        config.output_features = args.output_features

        config.data_path = args.data_path
        config.load_checkpoint = args.load_checkpoint

        config.batch_size = args.batch_size
        config.num_workers = args.num_workers
        config.pool_size = args.pool_size
        config.seed = args.seed

        config.max_epochs = args.max_epochs
        config.k_fold = args.k_fold

        config.exclusive_arch = args.exclusive_arch
        config.exclusive_opt = args.exclusive_opt

        config.redis = args.redis
        config.data_name = args.data_name
        config.mode = args.mode

    if args.max_epochs != 200:
        config.max_epochs = args.max_epochs

    if args.exclusive_arch is not None:
        config.exclusive_arch = args.exclusive_arch

    if args.exclusive_opt is not None:
        config.exclusive_opt = args.exclusive_opt

    if args.k_fold != 0:
        config.k_fold = args.k_fold

    if args.mode != "file":
        config.mode = args.mode

    return config


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    config: TrainConfig = read_config()
    random_seed = config.seed
    seed_everything(random_seed)

    print("Loading Dataset......")
    pool_size = config.pool_size
    if config.redis:
        my_dataset = ASTGraphRedisDataModule(
            data_name=config.data_name,
            pool_size=pool_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            k_fold=config.k_fold,
            data_path=config.data_path,
        )
        my_dataset.prepare_data()
    else:
        my_dataset = ASTGraphDataModule(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_path=config.data_path,
            pool_size=pool_size,
            k_fold=config.k_fold,
            exclusive_arch=config.exclusive_arch,
            exclusive_opt=config.exclusive_opt,
        )
        my_dataset.prepare_data()

        print(
            "Dataset Loaded. adj length:",
            my_dataset.max_length,
            "feature length:",
            my_dataset.feature_length,
        )

    load_checkpoint = config.load_checkpoint
    if load_checkpoint:
        print("Loading Checkpoint......")
        my_model = PLModelForAST(
            adj_length=my_dataset.max_length,
            in_features=my_dataset.feature_length,
            lr=4e-3,
            pool_size=pool_size,
            alpha=0.2,
            dropout=0.3,
            hidden_features=64,
            n_heads=6,
            output_features=128,
            seed=random_seed,
            data_path=config.data_path + str(config.k_fold),
        ).load_from_checkpoint(load_checkpoint)
        print("Checkpoint Loaded.")
    else:
        my_model = PLModelForAST(
            adj_length=my_dataset.max_length,
            in_features=my_dataset.feature_length,
            lr=4e-3,
            pool_size=pool_size,
            alpha=0.2,
            dropout=0.3,
            hidden_features=64,
            n_heads=6,
            output_features=128,
            seed=random_seed,
            data_path=config.data_path + str(config.k_fold),
        )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss_all",
        mode="min",
        save_on_train_epoch_end=True,
        save_last=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp_fork",
        precision="16-mixed",
        max_epochs=config.max_epochs,
        # val_check_interval=0.3,
        callbacks=[checkpoint_callback],
        # logger=None
        # gradient_clip_val=0.5,
    )

    trainer.fit(
        model=my_model,
        train_dataloaders=my_dataset,
    )
