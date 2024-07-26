import os
import random
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
import threading
from sklearn.model_selection import KFold

from FileScanner import FileTree
from GraphConverter import Converter

import dgl
import torch
import json


def purify_cpg_json(cpg_json: dict):
    return {
        "adj": cpg_json["adj"],
        "feature": cpg_json["feature"],
        "name": cpg_json["name"],
    }


def split_train_test_set(all_data: dict, ratio=0.1):
    if len(all_data) > 2:
        binary_list = list(all_data.keys())
        test_name = random.sample(binary_list, int(len(binary_list) * ratio))
        train_set, test_set = split_dataset_based_on_name(
            all_data, test_name, binary_level=True
        )

    else:
        # It means it's the function mode
        binary_name = list(all_data.keys())[0]
        function_list = list(all_data[binary_name].keys())
        test_name = random.sample(function_list, int(len(function_list) * ratio))
        train_set, test_set = split_dataset_based_on_name(
            all_data, test_name, binary_level=False
        )

    return train_set, test_set


def split_dataset_based_on_name(all_data: dict, test_name: list, binary_level=True):
    if binary_level:
        train_set = {}
        test_set = {}

        for binary in all_data:
            if binary in test_name:
                test_set[binary] = all_data[binary]
            else:
                train_set[binary] = all_data[binary]

        return train_set, test_set

    else:
        binary_name = list(all_data.keys())[0]
        train_set = {binary_name: {}}
        test_set = {binary_name: {}}

        for function_name in all_data[binary_name]:
            if function_name in test_name:
                test_set[binary_name][function_name] = all_data[binary_name][
                    function_name
                ]
            else:
                train_set[binary_name][function_name] = all_data[binary_name][
                    function_name
                ]

        return train_set, test_set


def KFold_split(all_data: dict, K=10):
    if len(all_data) > 2:
        # do the K Fold by binary name
        binary_list = list(all_data.keys())
        KFold_splitter = KFold(n_splits=K, shuffle=True)
        k_folds = KFold_splitter.split(binary_list)
        for train_name, test_name in k_folds:
            test_name = np.array(binary_list)[test_name]
            train_set, test_set = split_dataset_based_on_name(
                all_data, test_name, binary_level=True
            )
            yield train_set, test_set

    else:
        binary_name = list(all_data.keys())[0]
        function_list = list(all_data[binary_name].keys())
        KFold_splitter = KFold(n_splits=K, shuffle=True)
        k_folds = KFold_splitter.split(function_list)
        for train_name, test_name in k_folds:
            test_name = np.array(function_list)[test_name]
            train_set, test_set = split_dataset_based_on_name(
                all_data, test_name, binary_level=False
            )
            yield train_set, test_set


def purity_dataset(all_data: dict):
    for binary in all_data:
        for function in all_data[binary]:
            temp = []
            for cpg_json in all_data[binary][function]:
                temp.append(purify_cpg_json(cpg_json))
            all_data[binary][function] = temp.copy()

    return all_data


def filter_dataset(all_data: dict):
    bad_funtion = []
    for binary in all_data:
        for function in all_data[binary]:
            if len(all_data[binary][function]) < 2:
                bad_funtion.append((binary, function))

    for binary, function in bad_funtion:
        del all_data[binary][function]

    bad_binary = []
    for binary in all_data:
        if len(all_data[binary]) < 2:
            bad_binary.append(binary)

    for binary in bad_binary:
        del all_data[binary]

    return all_data


def filter_bad_dataset(all_data: dict):
    bad_index = []
    bad_funtion = []
    for binary in all_data:
        for function in all_data[binary]:
            if len(all_data[binary][function]) < 2:
                bad_funtion.append((binary, function))

    for binary, function in bad_funtion:
        for function_body in all_data[binary][function]:
            bad_index.append(function_body["index"])
        del all_data[binary][function]

    bad_binary = []
    for binary in all_data:
        if len(all_data[binary]) < 2:
            bad_binary.append(binary)

    for binary in bad_binary:

        for function in all_data[binary]:
            for function_body in all_data[binary][function]:
                bad_index.append(function_body["index"])

        del all_data[binary]

    bad_index = sorted(list(set(bad_index)), reverse=True)

    return all_data, bad_index


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)
        f.close()


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        obj = pickle.load(f)
        f.close()
    return obj


def dataset_size(dataset: dict):
    size = 0
    for binary in dataset:
        for function in dataset[binary]:
            size += len(dataset[binary][function])
    return size


def apply_dgl(
    dataset: dict, save_path: str, max_length: int = 0, add_nodes: bool = False
):
    file_index = {}
    dgl_array = []

    dgl_path = os.path.join(save_path, "dgl_graphs.dgl")
    index_path = os.path.join(save_path, "total_index.pkl")

    is_bad = False
    pbar = tqdm(total=dataset_size(dataset))
    pbar.set_description("Converting the dataset to dgl")

    for binary_name in dataset:
        if binary_name not in file_index:
            file_index[binary_name] = {}

        for function_name in dataset[binary_name]:
            if function_name not in file_index[binary_name]:
                file_index[binary_name][function_name] = []

            for function_body in dataset[binary_name][function_name]:
                graph = dgl.graph((function_body["adj"][0], function_body["adj"][1]))
                graph = dgl.to_bidirected(graph)
                graph = dgl.add_self_loop(graph)
                try:
                    graph.ndata["feat"] = torch.tensor(
                        function_body["feature"], dtype=torch.float
                    )
                except dgl._ffi.base.DGLError:
                    print(
                        f"Bad Function detected from {binary_name} - {function_name}, need to more sanitize"
                    )
                    is_bad = True
                    pbar.update()
                    continue
                del function_body["adj"]
                del function_body["feature"]

                if add_nodes:
                    padding_length = max_length - graph.number_of_nodes()
                    graph = dgl.add_nodes(graph, padding_length)
                    graph = dgl.add_self_loop(graph)
                # padding_length = max_length - graph.number_of_nodes()
                # graph = dgl.add_nodes(graph, padding_length)

                index = len(dgl_array)
                dgl_array.append(graph)

                function_body["index"] = index
                file_index[binary_name][function_name].append(function_body)
                pbar.update()

    pbar.close()

    if is_bad:
        file_index, bad_index = filter_bad_dataset(file_index)
        print("bad index length: ", len(bad_index))

    print(
        f"Finish converting, saving the dgl to {dgl_path} and index file to {index_path}"
    )
    dgl.save_graphs(dgl_path, dgl_array)
    with open(index_path, "wb") as f:
        pickle.dump(file_index, f)
        f.close()
    return file_index


class DataGenerator:
    def __init__(
        self,
        file_tree: FileTree,
        converter: Converter,
        save_path: str,
        read_cache: bool = False,
    ):
        self.file_tree = file_tree
        self.converter = converter
        self.save_path = save_path
        self.read_cache = read_cache

    def operator_walker(self):
        missing_operator = set()
        pbar = tqdm(total=len(self.file_tree))
        for file_path, _, _, _ in self.file_tree:
            for c_dot_file in os.listdir(file_path):
                pbar.update()
                full_path = os.path.join(file_path, c_dot_file)
                try:
                    self.converter.probe_operator_file(full_path)
                except UserWarning as e:
                    missing_operator.add(str(e))

        return list(missing_operator)

    def generate_all_data(self):
        all_data = {}
        pbar = tqdm(total=len(self.file_tree))
        for file_path, opt, arch, binary in self.file_tree:
            if binary not in all_data:
                all_data[binary] = {}
            for c_dot_file in os.listdir(file_path):
                pbar.update()
                c_dot_filename = os.path.join(file_path, c_dot_file)
                cpg_json = self.converter.convert_file(
                    c_dot_filename, binary_name=binary, opt=opt, arch=arch
                )

                if not cpg_json:
                    continue

                function_name = cpg_json["name"]
                if function_name not in all_data[binary]:
                    all_data[binary][function_name] = []
                all_data[binary][function_name].append(cpg_json)

        return all_data

    def wrap_dataset(self, dataset):
        return {
            "data": dataset,
            "adj": self.converter.max_length,
            "feature_len": self.converter.length,
        }

    def run(self, k_fold: int = 0):

        if self.converter.read_op:
            self.converter.load_op_list(self.converter.op_file)
        else:
            print("Start to scan operator...")
            missing = self.operator_walker()
            print("adding missing operator: ", missing, "to OP")
            self.converter.add_op_to_list(missing)
            self.converter.save_op_list(self.converter.op_file)
            self.converter.load_op_list(self.converter.op_file)

        if self.read_cache and os.path.exists(
            os.path.join(self.save_path, "origin_data.pkl")
        ):
            print("Reading Cached dataset")
            all_data = load_pickle(os.path.join(self.save_path, "origin_data.pkl"))
        else:
            print("Generating dataset...")
            all_data = self.generate_all_data()

            print("Saving the original dataset...")
            save_pickle(all_data, os.path.join(self.save_path, "origin_data.pkl"))

        print("filtering lonely dataset...")
        all_data = filter_dataset(all_data)
        all_data = apply_dgl(
            all_data, self.save_path, adj_len=self.converter.max_length
        )

        if k_fold:
            print("Splitting dataset...")
            for i, (train_data, test_data) in enumerate(
                KFold_split(all_data, K=k_fold)
            ):
                batch = i + 1
                print("saving dataset {}...".format(batch))
                save_pickle(
                    self.wrap_dataset(train_data),
                    os.path.join(
                        self.save_path, "index_train_data_{}.pkl".format(batch)
                    ),
                )
                save_pickle(
                    self.wrap_dataset(test_data),
                    os.path.join(
                        self.save_path, "index_test_data_{}.pkl".format(batch)
                    ),
                )
        else:
            # return
            print("Splitting dataset...")
            train_data, test_data = split_train_test_set(all_data)

            print("saving dataset...")
            save_pickle(
                self.wrap_dataset(train_data),
                os.path.join(self.save_path, "index_train_data.pkl"),
            )
            save_pickle(
                self.wrap_dataset(test_data),
                os.path.join(self.save_path, "index_test_data.pkl"),
            )


class DataGeneratorMultiProcessing(DataGenerator):
    def __init__(
        self,
        file_tree: FileTree,
        converter: Converter,
        save_path: str,
        cores: int = multiprocessing.cpu_count(),
        read_cache: bool = False,
    ):
        super().__init__(file_tree, converter, save_path, read_cache)
        self.cores = cores
        self.op_queue = multiprocessing.Manager().Queue(self.cores * 32)
        self.convert_queue = multiprocessing.Manager().Queue(self.cores * 32)

        self.is_finish: multiprocessing.Queue = multiprocessing.Manager().Queue(1)

    def generate_data_reduce(self):
        all_data = {}

        while True:
            if (not self.is_finish.empty()) and self.convert_queue.empty():
                self.is_finish.get()
                break
            if self.convert_queue.empty():
                continue
            cpg_json, binary_name = self.convert_queue.get()

            function_name = cpg_json["name"]

            if binary_name not in all_data:
                all_data[binary_name] = {}

            if function_name not in all_data[binary_name]:
                all_data[binary_name][function_name] = []

            all_data[binary_name][function_name].append(cpg_json)

        print("Finish reducing, saving the pkl file...")
        save_pickle(all_data, os.path.join(self.save_path, "origin_data.pkl"))

    def generate_data_map(self, file_path, opt, arch, binary, symbol_table: dict = {}):
        cpg_json = self.converter.convert_file(
            file_path, binary_name=binary, opt=opt, arch=arch
        )
        if not cpg_json:
            return None
        else:
            self.convert_queue.put((cpg_json, binary))

    def generate_all_data(self):

        while not self.is_finish.empty():
            self.is_finish.get()

        print("totally {} files".format(len(self.file_tree)))

        pbar = tqdm(total=len(self.file_tree))
        pbar.set_description("Converting the dot file")

        def update(*args):
            pbar.update()

        reduce_pool = multiprocessing.Pool(1)
        reduce_pool.apply_async(self.generate_data_reduce)

        with multiprocessing.Pool(self.cores - 1) as pool:
            for file_path, opt, arch, binary in self.file_tree:

                for c_dot_file in os.listdir(file_path):
                    c_dot_filename = os.path.join(file_path, c_dot_file)
                    pool.apply_async(
                        self.generate_data_map,
                        args=(c_dot_filename, opt, arch, binary),
                        callback=update,
                    )

            pool.close()
            pool.join()
        self.is_finish.put(True)

        # print("Waiting for reducer to finish...")
        # all_data = reduce_result.get()
        reduce_pool.close()
        reduce_pool.join()

        pbar.close()

    def operator_walker_map(self, file_path):
        try:
            self.converter.probe_operator_file(file_path)
        except UserWarning as e:
            self.op_queue.put(str(e))

    def operator_walker_reduce(self):
        missing_op = set()
        while True:
            if not self.is_finish.empty() and self.op_queue.empty():
                self.is_finish.get()
                break
            if self.op_queue.empty():
                continue
            missing_op.add(self.op_queue.get())
        return missing_op

    def operator_walker(self):
        while not self.is_finish.empty():
            self.is_finish.get()

        pbar = tqdm(total=len(self.file_tree))
        pbar.set_description("Scanning the operator")

        def update(*args):
            pbar.update()

        reducer = multiprocessing.Pool(1).apply_async(self.operator_walker_reduce)

        with multiprocessing.Pool(self.cores - 1) as pool:

            for file_path, _, _, _ in self.file_tree:
                for c_dot_file in os.listdir(file_path):
                    full_path = os.path.join(file_path, c_dot_file)
                    pool.apply_async(
                        self.operator_walker_map, args=(full_path,), callback=update
                    )

            pool.close()
            pool.join()

        self.is_finish.put(True)

        print("Waiting for reducer to finish...")
        missing_op = reducer.get()
        pbar.close()

        origin_operator = self.converter.OP
        now_operator = origin_operator + list(missing_op)
        self.converter.OP = now_operator
        self.converter.save_op_list(self.converter.op_file)
        return list(missing_op)

    def run(self, k_fold: int = 0, add_nodes: bool = False):

        if self.converter.read_op and os.path.exists(self.converter.op_file):
            self.converter.load_op_list(self.converter.op_file)
        else:
            print("Start to scan operator...")
            missing = self.operator_walker()
            print("adding missing operator: ", missing, "to OP")
            self.converter.add_op_to_list(missing)
            self.converter.save_op_list(self.converter.op_file)
            self.converter.load_op_list(self.converter.op_file)

        if self.read_cache and os.path.exists(
            os.path.join(self.save_path, "origin_data.pkl")
        ):
            print("Reading Cached dataset")
            all_data = load_pickle(os.path.join(self.save_path, "origin_data.pkl"))
        else:
            print("Generating dataset...")
            self.generate_all_data()

            print("Saving the original dataset...")
            all_data = load_pickle(os.path.join(self.save_path, "origin_data.pkl"))

        all_data = filter_dataset(all_data)
        data_index = apply_dgl(
            all_data, self.save_path, self.converter.max_length, add_nodes=add_nodes
        )
        save_pickle(
            self.wrap_dataset(data_index), os.path.join(self.save_path, "index_all.pkl")
        )

        if k_fold:
            print("Splitting dataset...")
            for i, (train_data, test_data) in enumerate(
                KFold_split(data_index, K=k_fold)
            ):
                batch = i + 1
                print("saving dataset {}...".format(batch))
                save_pickle(
                    self.wrap_dataset(train_data),
                    os.path.join(
                        self.save_path, "index_train_data_{}.pkl".format(batch)
                    ),
                )
                save_pickle(
                    self.wrap_dataset(test_data),
                    os.path.join(
                        self.save_path, "index_test_data_{}.pkl".format(batch)
                    ),
                )
        else:
            print("Splitting dataset...")
            train_data, test_data = split_train_test_set(data_index)

            print("saving dataset...")
            # save_pickle(self.wrap_dataset(all_data), os.path.join(self.save_path, 'all_data.pkl'))
            save_pickle(
                self.wrap_dataset(train_data),
                os.path.join(self.save_path, "index_train_data.pkl"),
            )
            save_pickle(
                self.wrap_dataset(test_data),
                os.path.join(self.save_path, "index_test_data.pkl"),
            )
