import copy
import json
import os
import pickle
import random
import time
# import pytorch_lightning as pl
from typing import List, Union, Dict

import dgl
import lightning.pytorch as pl
import psutil
import redis
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

FunctionBody = Dict[str, Union[int, float, str]]
DataIndex = Dict[str, Dict[str, List[FunctionBody]]]


class ASTGraphDataset(Dataset):
    def __init__(
            self, data_path: str, data_index: DataIndex, max_adj: int, feature_len: int, pool_size: int, mode: str,
            environment: tuple = None
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_index: DataIndex = data_index
        self.binary_list = list(self.data_index.keys())
        self.phases = [0]
        self.length = self._get_length()
        self.max_adj = max_adj
        self.feature_len = feature_len
        self.pool_size = pool_size
        self.mode = mode
        self.environment = environment

    def __len__(self):
        return self.length

    def _get_length(self):
        length = 0
        phases = [0]
        for key in self.binary_list:
            length += len(self.data_index[key])
            phases.append(phases[-1] + len(self.data_index[key]))
        self.phases = phases
        return length

    def _find_binary_index(self, index):
        for i in range(len(self.phases)):
            if index < self.phases[i]:
                return i - 1, index - self.phases[i - 1]
        return -1, -1

    # @profile
    def __getitem__(self, index):
        # impliment the sliced file reading

        binary_index, function_offset = self._find_binary_index(index)
        binary_name = self.binary_list[binary_index]
        function_name = sorted(list(self.data_index[binary_name].keys()))[function_offset]
        sample_function_list = self.data_index[binary_name][function_name]

        same_pair = random.sample(sample_function_list, 2)
        sample, same_sample = same_pair[0], same_pair[1]

        if self.mode == 'file':

            different_binary_name = random.choice(self.binary_list)
            different_function_name = random.choice(
                list(self.data_index[different_binary_name].keys())
            )
        
            while (
                    different_function_name == function_name
                    and different_binary_name == binary_name
            ):
                different_binary_name = random.choice(self.binary_list)
                different_function_name = random.choice(
                    list(self.data_index[different_binary_name].keys())
                )

            different_sample = random.choice(
                self.data_index[different_binary_name][different_function_name]
            )
        
        else:
            arch = sample["arch"]
            opt = sample["opt"]
            different_sample = sample
            while arch == sample["arch"] or opt == sample["opt"]:
                random_binary_name = random.choice(self.binary_list)
                random_function_name = random.choice(list(self.data_index[random_binary_name].keys()))
                different_sample = random.choice(self.data_index[random_binary_name][random_function_name])

        different_sample = self._to_tensor(different_sample)
        sample_dict = sample
        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)

        # Pool candidates
        if self.pool_size:
            pool = self._get_pool(sample=sample_dict)
            pool = self._many_to_tensor(pool)
            return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample,
                    "label": torch.tensor([0]), "pool": pool}

        return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample,
                "label": torch.tensor([0])}

    def _get_pool(self, sample: dict):
        pool = []
        # Get the function pool that does not contain the function_name
        for p in range(self.pool_size):
            pool_binary_name = random.choice(self.binary_list)
            pool_function_name = random.choice(list(self.data_index[pool_binary_name].keys()))
            
            pool_item = random.choice(self.data_index[pool_binary_name][pool_function_name])
            
            if self.mode == "file":  
                while (
                        pool_binary_name == sample['binary'] and pool_function_name == sample['name']
                ):
                    pool_binary_name = random.choice(self.binary_list)
                    pool_function_name = random.choice(
                        list(self.data_index[pool_binary_name].keys())
                    )
                    pool_item = random.choice(self.data_index[pool_binary_name][pool_function_name])
            
            else:
                random_arch, random_opt = sample["arch"], sample["opt"]
                pool_item = sample
                while (random_arch == sample["arch"] and random_opt == sample["opt"]):
                    pool_binary_name = random.choice(self.binary_list)
                    pool_function_name = random.choice(list(self.data_index[pool_binary_name].keys()))
                    pool_item = random.choice(self.data_index[pool_binary_name][pool_function_name])
                    random_arch, random_opt = pool_item["arch"], pool_item["opt"]
                    
            if self.environment:
                functions = self.data_index[pool_binary_name][pool_function_name]
                for func in functions:
                    if func["arch"] == self.environment[0] and func["opt"] == self.environment[1]:
                        pool.append(func)
                        break
                pool_item = random.choice(functions)
            else:
                pool.append(pool_item)

        return pool

    def _get_env_pool(self, binary_name: str, function_name: str):
        pool = []
        for p in range(self.pool_size):
            pool_binary_name = random.choice(self.binary_list)
            pool_function_name = random.choice(list(self.data_index[pool_binary_name].keys()))
            while (
                    pool_binary_name == binary_name and pool_function_name == function_name
            ):
                pool_binary_name = random.choice(self.binary_list)
                pool_function_name = random.choice(
                    list(self.data_index[pool_binary_name].keys())
                )
            pool.append(random.choice(self.data_index[pool_binary_name][pool_function_name]))

        return pool

    # @profile
    def _to_tensor(self, data: dict):
        index = data['index']
        graph: dgl.DGLGraph = dgl.load_graphs(self.data_path, idx_list=[index])[0][0]

        if graph.number_of_nodes() < self.max_adj:
            padding_size = self.max_adj - graph.number_of_nodes()
            graph = dgl.add_nodes(graph, padding_size)
            graph = dgl.add_self_loop(graph)

        return graph

    def _many_to_tensor(self, data: List[dict]):
        indices = [x['index'] for x in data]
        graphs = dgl.load_graphs(self.data_path, idx_list=indices)[0]
        for i in range(len(graphs)):
            if graphs[i].number_of_nodes() < self.max_adj:
                padding_size = self.max_adj - graphs[i].number_of_nodes()
                graphs[i] = dgl.add_nodes(graphs[i], padding_size)
                graphs[i] = dgl.add_self_loop(graphs[i])
        return graphs


def collate_fn(x):
    batch_size = len(x)
    sample_list = []
    same_sample_list = []
    different_sample_list = []
    for i in range(batch_size):
        sample_list.append(x[i]["sample"])
        same_sample_list.append(x[i]["same_sample"])
        different_sample_list.append(x[i]["different_sample"])
    sample_list = dgl.batch(sample_list)
    same_sample_list = dgl.batch(same_sample_list)
    different_sample_list = dgl.batch(different_sample_list)

    if "pool" in x[0]:
        batch_list = []
        for i in range(batch_size):
            pool_list = x[i]["pool"]
            pool_list = dgl.batch(pool_list)
            batch_list.append(pool_list)
        return {"sample": sample_list, "same_sample": same_sample_list, "different_sample": different_sample_list,
                "label": torch.tensor([0]), "pool": batch_list}
    return {"sample": sample_list, "same_sample": same_sample_list, "different_sample": different_sample_list,
            "label": torch.tensor([0])}


def _load_pickle_data(data_path: str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        f.close()
    feature_len = data["feature_len"]
    adj_len = data["adj"]
    data = data["data"]
    return adj_len, feature_len, data


class ASTGraphRedisDataset(ASTGraphDataset):
    def __init__(self,
                 data_name: str,
                 redis_connect: redis.ConnectionPool,
                 data_index: DataIndex,
                 pool_size: int,
                 ):
        super().__init__(data=[], data_index=data_index, max_adj=-1, feature_len=-1, pool_size=pool_size)
        self.redis = redis.Redis(connection_pool=redis_connect)
        self.data_name = data_name

    def _to_tensor(self, data: dict):
        index = data['index']
        name = self.data_name + '-' + str(index)
        graph_bytes: bytes = self.redis.get(name)
        graph: dgl.DGLGraph = pickle.loads(graph_bytes)
        return graph

    def _to_tensor_list(self, data: list):
        indices = [x['index'] for x in data]
        names = [self.data_name + '-' + str(x) for x in indices]
        graph_bytes_list: List[bytes] = self.redis.mget(names)
        graph_list: List[dgl.DGLGraph] = [pickle.loads(x) for x in graph_bytes_list]
        return graph_list

    def __getitem__(self, index):
        # impliment the sliced file reading

        binary_index, function_offset = self._find_binary_index(index)
        binary_name = self.binary_list[binary_index]
        function_name = sorted(list(self.data_index[binary_name].keys()))[function_offset]
        sample_function_list = self.data_index[binary_name][function_name]

        same_pair = random.sample(sample_function_list, 2)
        sample, same_sample = same_pair[0], same_pair[1]

        different_binary_name = random.choice(self.binary_list)
        different_function_name = random.choice(
            list(self.data_index[different_binary_name].keys())
        )

        while (
                different_function_name == function_name
                and different_binary_name == binary_name
        ):
            different_binary_name = random.choice(self.binary_list)
            different_function_name = random.choice(
                list(self.data_index[different_binary_name].keys())
            )

        different_sample = random.choice(
            self.data_index[different_binary_name][different_function_name]
        )

        sample = self._to_tensor(sample)
        same_sample = self._to_tensor(same_sample)
        different_sample = self._to_tensor(different_sample)

        # Pool candidates
        if self.pool_size:
            pool = self._get_pool(binary_name, function_name)
            pool = self._to_tensor_list(pool)
            return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample,
                    "label": torch.tensor([0]), "pool": pool}

        return {"sample": sample, "same_sample": same_sample, "different_sample": different_sample,
                "label": torch.tensor([0])}


class ASTGraphDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "!pairs.pkl",
            pool_size: int = 0,
            batch_size: int = 32,
            num_workers: int = 16,
            mode: str = "file",
            exclude: list = None,
            k_fold: int = 0,
            exclusive_arch: str = None,
            exclusive_opt: str = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path

        self.train_set: Union[Dataset, None] = None
        self.val_set: Union[Dataset, None] = None

        self.pool_size = pool_size

        self.exclude = exclude

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.max_length = -1
        self.feature_length = -1

        self.k_fold = k_fold

        self.mode = mode

        if exclusive_arch and exclusive_opt:
            self.environment = (exclusive_arch, exclusive_opt)
        else:
            self.environment = None

    def filter_environment_for_train(self, data_dict: dict, environment: tuple):
        # Need to modify to support the binary mode
        target_arch, target_opt = environment
        filtered_dict = copy.deepcopy(data_dict)
        for binary_name in data_dict:
            for function_name in data_dict[binary_name]:
                filtered_dict[binary_name][function_name] = [x for x in filtered_dict[binary_name][function_name] if
                                                             x["arch"] != target_arch or x["opt"] != target_opt]

        for binary_name in filtered_dict:
            bad_function_name_list = []
            for function_name in filtered_dict[binary_name]:
                if len(filtered_dict[binary_name][function_name]) < 2:
                    bad_function_name_list.append(function_name)
            for bad_function_name in bad_function_name_list:
                del filtered_dict[binary_name][bad_function_name]

        bad_binary_name_list = []
        for binary_name in filtered_dict:
            if len(filtered_dict[binary_name]) < 2:
                bad_binary_name_list.append(binary_name)
        for bad_binary_name in bad_binary_name_list:
            del filtered_dict[bad_binary_name]

        return filtered_dict

    def filter_environment_for_test(self, data_dict: dict, environment: tuple):
        target_arch, target_opt = environment
        # Pick out the functions that don't contain the target environment
        # Need to modify to support the binary mode

        for binary_name in data_dict:
            kick_function_name_list = []
            for function_name in data_dict[binary_name]:
                is_kick = True
                for function_body in data_dict[binary_name][function_name]:
                    if function_body["arch"] == target_arch and function_body["opt"] == target_opt:
                        is_kick = False
                if is_kick:
                    kick_function_name_list.append(function_name)

            for kick_function_name in kick_function_name_list:
                del data_dict[binary_name][kick_function_name]

        kick_binary_name_list = []
        for binary_name in data_dict:
            if len(data_dict[binary_name]) < 2:
                kick_binary_name_list.append(binary_name)
        for kick_binary_name in kick_binary_name_list:
            del data_dict[kick_binary_name]

        for binary_name in data_dict:
            assert len(data_dict[
                           binary_name]) >= self.pool_size + 1, f"Binary {binary_name} has less than {self.pool_size + 1} functions"
        return data_dict

    def prepare_data(self):
        if self.k_fold:
            train_path = os.path.join(
                self.data_path, f"index_train_data_{self.k_fold}.pkl"
            )
            test_path = os.path.join(
                self.data_path, f"index_test_data_{self.k_fold}.pkl"
            )
        else:
            train_path = os.path.join(self.data_path, "index_train_data.pkl")
            test_path = os.path.join(self.data_path, "index_test_data.pkl")

        adj_len, feature_len, index_train_data = _load_pickle_data(train_path)

        # Assume train_set and test_set are the same
        _, _, index_test_data = _load_pickle_data(test_path)

        if self.environment:
            index_train_data = self.filter_environment_for_train(index_train_data, self.environment)
            index_test_data = self.filter_environment_for_test(index_test_data, self.environment)

        self.max_length = adj_len
        self.feature_length = feature_len

        dgl_path = os.path.join(self.data_path, "dgl_graphs.dgl")

        self.train_set = ASTGraphDataset(
            data_path=dgl_path,
            data_index=index_train_data,
            max_adj=self.max_length,
            feature_len=self.feature_length,
            pool_size=self.pool_size,
            mode=self.mode,
        )

        self.val_set = ASTGraphDataset(
            data_path=dgl_path,
            data_index=index_test_data,
            max_adj=self.max_length,
            feature_len=self.feature_length,
            pool_size=self.pool_size,
            mode=self.mode,
            environment=self.environment
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=8
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            # prefetch_factor=16
        )


class ASTGraphRedisDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_name: str = "uboot",
            pool_size: int = 0,
            batch_size: int = 32,
            num_workers: int = 16,
            k_fold: int = 0,
            data_path: str = "dataset/uboot_dataset",

    ):
        super().__init__()

        self.train_set: Union[Dataset, None] = None
        self.val_set: Union[Dataset, None] = None

        self.data_name = data_name
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.redis = redis.ConnectionPool(host="localhost", port=6379, db=0)
        self.k_fold = k_fold
        self.data_path = data_path

    def prepare_data(self):
        if self.k_fold:
            train_path = os.path.join(
                self.data_path, f"index_train_data_{self.k_fold}.pkl"
            )
            test_path = os.path.join(
                self.data_path, f"index_test_data_{self.k_fold}.pkl"
            )
        else:
            train_path = os.path.join(self.data_path, "index_train_data.pkl")
            test_path = os.path.join(self.data_path, "index_test_data.pkl")

        adj_len, feature_len, index_train_data = _load_pickle_data(train_path)

        # Assume train_set and test_set are the same
        _, _, index_test_data = _load_pickle_data(test_path)

        self.max_length = adj_len
        self.feature_length = feature_len

        self.train_set = ASTGraphRedisDataset(
            data_name=self.data_name,
            redis_connect=self.redis,
            data_index=index_train_data,
            pool_size=self.pool_size,
        )

        self.val_set = ASTGraphRedisDataset(
            data_name=self.data_name,
            redis_connect=self.redis,
            data_index=index_test_data,
            pool_size=self.pool_size,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
            # prefetch_factor=16
        )


if __name__ == "__main__":
    a0 = time.time()
    p = ASTGraphRedisDataModule(data_name="uboot", pool_size=50, batch_size=4, num_workers=8, k_fold=1,)
    # p = ASTGraphDataModule(data_path="dataset/uboot_dataset", pool_size=50, batch_size=10, num_workers=4, k_fold=1)
    p.prepare_data()
    train = p.train_dataloader()
    idx = 0
    a1 = time.time()
    print("Overhead: ", a1 - a0)
    print(
        "当前进程的内存使用：%.4f GB"
        % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    )
    a2 = a1
    for i in train:
        idx += 1
        print(idx)
        print("Single_time:", time.time() - a2)
        a2 = time.time()
        # print(i)
        # break
    a3 = time.time()
    print("总共用时：", a3 - a0)
