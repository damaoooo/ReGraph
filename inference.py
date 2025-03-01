import random
import torch
import os
import numpy as np
import pickle
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
from model import PLModelForAST, pearson_score
from dataset import ASTGraphDataModule

from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score

import dgl


def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], v2.T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    res = num / denom
    return 0.5 + 0.5 * res


def similarity_score(query, vectors):
    distance = np.linalg.norm(query - vectors, axis=-1)
    score = 1 / (1 + distance)
    return score


def get_pearson_score(query, vectors):
    cov = (query * vectors).mean(axis=-1)
    pearson = cov / (query.std(axis=-1) * vectors.std(axis=-1))
    return abs(pearson), pearson


class FunctionEmbedding:
    def __init__(self, name: str, embedding):
        self.name = name
        self.embedding = embedding
        self.cosine = 0


class ModelConfig:
    def __init__(self):
        # file location
        self.model_path = (
            "lightning_logs/c_re_door/checkpoints/epoch=59-step=42554.ckpt"
        )
        self.dataset_path = "./c_language"
        self.exclude_list = []

        # Runtime Setting
        self.cuda = False

        # If no dataset is provided
        self.feature_length = 139
        self.max_length = 1000

        # Model Hyper-Parameters
        self.alpha = 0.2
        self.dropout = 0.3
        self.hidden_features = 64
        self.n_heads = 6
        self.output_features = 128

        # Predict
        self.topK = 3


class InferenceModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
        if self.config.dataset_path:
            total_path = os.path.join(self.config.dataset_path, "!total.pkl")
            self.dataset = ASTGraphDataModule(
                total_path, exclude=self.config.exclude_list, batch_size=1
            )
            self.dataset.prepare_data()
            self.feature_length = self.dataset.feature_length
            self.max_length = self.dataset.max_length
        else:
            self.feature_length = self.config.feature_length
            self.max_length = self.config.max_length
            self.dataset = None

        self.model = PLModelForAST.load_from_checkpoint(
            self.config.model_path, strict=False
        )

        if self.config.cuda:
            self.device_name = "cuda"
        else:
            self.device_name = "cpu"

        if self.config.cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model = torch.compile(self.model)
        self.model.eval()

    def single_dgl_to_embedding(self, data: dgl.DGLGraph):
        with torch.no_grad():

            # padding = self.max_length - data.num_nodes()
            # data = dgl.add_nodes(data, padding)
            # data = dgl.add_self_loop(data)

            if self.config.cuda:
                data = data.to("cuda")
            tembedding = self.model.my_model(data)
            tembedding = tembedding.detach().cpu()

            embedding = tembedding.numpy()
            del tembedding

            return embedding

    def judge(self, func: str, candidate: list):
        if func in candidate:
            return True
        return False

    def merge_dgl_dict(self, dataset: dict, graphs: List[dgl.DGLGraph]):
        # Collect all graphs and their positions to be processed
        tasks = []
        positions = []
        for binary in dataset['data']:
            for function_name in dataset['data'][binary]:
                for i in range(len(dataset['data'][binary][function_name])):
                    index = dataset['data'][binary][function_name][i]['index']
                    tasks.append(graphs[index])
                    positions.append((binary, function_name, i))

        # Set up progress bar
        pbar = tqdm(total=len(tasks))
        pbar.set_description("Padding Graphs")
        
        # Process all graphs serially
        prepared_graphs = {}
        for i, graph in enumerate(tasks):
            # Process a single graph
            padding = self.max_length - graph.num_nodes()
            graph = dgl.add_nodes(graph, padding)
            graph = dgl.add_self_loop(graph)
            prepared_graphs[i] = graph
            pbar.update()
        
        pbar.close()

        print("Finished Converting... Fetching Data From Pool")

        pbar = tqdm(total=(len(tasks)))
        pbar.set_description("Inference")

        # Batch processing parameters
        batch_size = 256  # Adjust based on GPU memory size
        results = [None] * len(tasks)  # Pre-allocate results list
        
        # Process prepared graphs in batches
        for i in range(0, len(tasks), batch_size):
            batch_indices = range(i, min(i + batch_size, len(tasks)))
            batch_padding = []
            
            # Use already prepared graphs
            for idx in batch_indices:
                graph = prepared_graphs[idx]
                if self.config.cuda:
                    graph = graph.to('cuda')
                batch_padding.append(graph)
            
            # Batch processing
            if len(batch_padding) > 0:
                batched_graphs = dgl.batch(batch_padding)
                with torch.no_grad():
                    embeddings = self.model.my_model(batched_graphs)
                    embeddings = embeddings.detach().cpu().numpy()
                    # Unbatch and store results
                    for idx, embedding in zip(batch_indices, np.split(embeddings, len(batch_padding))):
                        results[idx] = embedding
            
            # Periodically clear GPU cache
            if self.config.cuda:
                torch.cuda.empty_cache()

            # Update progress bar
            pbar.update(batch_size)

        # Put results back into the dataset
        for (binary, function_name, i), embedding in zip(positions, results):
            dataset['data'][binary][function_name][i]['embedding'] = embedding

        pbar.close()
        return dataset

    def test_strip_recall_K(
        self,
        dataset_origin: dict,
        dataset_strip: dict,
        graph_strip: list,
        max_k: int = 10,
        n_candidates: int = 100,
        mode: str = "pool",
        cache_path: str = "",
    ):
        # Dataset 2 simulates a harder comparison
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k

        if cache_path:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    dataset_strip = pickle.load(f)
                    f.close()
            else:
                dataset_strip = self.merge_dgl_dict(dataset_strip, graph_strip)
                with open(cache_path, "wb") as f:
                    pickle.dump(dataset_strip, f)
                    f.close()
        else:
            dataset_strip = self.merge_dgl_dict(dataset_strip, graph_strip)

        pbar = tqdm(total=self.get_dataset_function_num(dataset_strip))

        length = []
        temp_ref = []

        for binary in dataset_origin["data"]:
            record_total = {x: [0, 0] for x in range(1, max_k + 1)}
            if binary not in dataset_strip["data"]:
                continue
            candidate_pool, candidate_name_list = self.get_function_file_set(
                dataset=dataset_strip, binary_name=binary
            )

            function_list1 = dataset_origin["data"][binary].keys()
            for function_name in function_list1:
                pbar.update()

                if len(dataset_origin["data"][binary][function_name]) < 2:
                    continue

                if function_name.startswith("function_"):
                    continue

                if function_name not in dataset_strip["data"][binary]:
                    for k in range(1, max_k + 1):
                        record_total[k][1] += 1
                    continue

                if len(dataset_strip["data"][binary][function_name]) < 2:
                    continue

                for function_body in dataset_origin["data"][binary][function_name]:

                    left_function = random.choice(
                        dataset_strip["data"][binary][function_name]
                    )
                    right_function = left_function

                    random_count = 0
                    while (
                        right_function["opt"] == left_function["opt"]
                        or right_function["arch"] == left_function["arch"]
                    ):
                        right_function = random.choice(
                            dataset_strip["data"][binary][function_name]
                        )
                        random_count += 1
                        if random_count > 100:
                            break

                    if random_count > 100:
                        continue

                    left_embedding = left_function["embedding"]
                    right_embedding = right_function["embedding"]

                    arch, opt = function_body["arch"], function_body["opt"]
                    function_names = []
                    function_candidates = []
                    random_count = 0

                    for i in range(n_candidates):
                        arch, opt = random.choice(list(candidate_pool.keys()))
                        while (
                            arch == function_body["arch"] or opt == function_body["opt"]
                        ):
                            arch, opt = random.choice(list(candidate_pool.keys()))
                            random_count += 1
                            if random_count > 100:
                                break
                        if random_count > 100:
                            break
                        selected = random.choice(candidate_pool[(arch, opt)])
                        function_names.append(selected.name)
                        function_candidates.append(selected.embedding)

                    if random_count > 100:
                        # No right function, continue
                        continue

                    function_candidates.append(right_embedding)
                    function_names.append(function_name)

                    function_candidates = np.vstack(function_candidates)
                    length.append(len(function_candidates))

                    # mm = similarity_score(left_embedding, function_candidates)
                    mm, ref = get_pearson_score(left_embedding, function_candidates)
                    rank_list = sorted(
                        zip(mm.reshape(-1), function_names),
                        key=lambda x: x[0],
                        reverse=True,
                    )[: self.config.topK]
                    for k in range(1, max_k + 1):
                        is_correct = self.judge(
                            right_function["name"], [x[1] for x in rank_list[:k]]
                        )
                        record_total[k][0] += int(is_correct)
                        record_total[k][1] += 1

                        if k == 1:
                            temp_ref.append(rank_list[0][2])

            success_result = True
            for k in range(1, max_k + 1):
                if record_total[k][1] == 0:
                    success_result = False
                    break

            if not success_result:
                continue

            for k in range(1, max_k + 1):
                recall[k].append(record_total[k][0] / record_total[k][1])

        recall_avg = []
        for k in range(1, max_k + 1):
            recall_avg.append(np.mean(recall[k]))

        pbar.close()

        # print("recall_avg", recall_avg)
        print("Pearson", np.mean(temp_ref), np.std(temp_ref))
        return recall_avg, temp_ref

    # @profile
    def test_recall_K(
        self,
        dataset: dict,
        graph_path: str,
        max_k: int = 10,
        n_candidates: int = 100,
        mode: str = "pool",
        cache_path: str = "",
    ):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k

        if cache_path:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    dataset = pickle.load(f)
                    f.close()
            else:
                graph, _ = dgl.load_graphs(graph_path)
                dataset = self.merge_dgl_dict(dataset, graph)
                with open(cache_path, "wb") as f:
                    pickle.dump(dataset, f)
                    f.close()
        else:
            graph, _ = dgl.load_graphs(graph_path)
            dataset = self.merge_dgl_dict(dataset, graph)

        pbar = tqdm(total=self.get_dataset_function_num(dataset))

        length = []
        for binary in dataset["data"]:
            record_total = {x: [0, 0] for x in range(1, max_k + 1)}
            print("Generating Function Pool for {}".format(binary))
            candidate_pool, candidate_name_list = self.get_function_file_set(
                dataset=dataset, binary_name=binary
            )
            # return
            function_list1 = dataset["data"][binary].keys()
            for function_name in function_list1:
                for function_body in dataset["data"][binary][function_name]:
                    pbar.update()

                    if len(dataset["data"][binary][function_name]) < 2:
                        continue

                    arch, opt = function_body["arch"], function_body["opt"]

                    if (arch, opt) not in candidate_pool:
                        continue

                    name, query_embedding = function_body["name"], FunctionEmbedding(
                        name=function_body["name"], embedding=function_body["embedding"]
                    )
                    query_embedding: FunctionEmbedding
                    query_embedding = query_embedding.embedding

                    mat2 = []
                    name_list = []

                    itself = random.choice(dataset["data"][binary][function_name])
                    random_count = 0
                    while (itself["arch"], itself["opt"]) == (arch, opt):
                        itself = random.choice(dataset["data"][binary][function_name])
                        random_count += 1
                        if random_count > 100:
                            break

                    if random_count > 100:
                        continue

                    # mat2.append(function_body['embedding'])
                    if mode == "file":
                        random_arch = (itself["arch"], itself["opt"])

                        for c in candidate_pool[random_arch]:
                            if c.name == name:
                                continue
                            c: FunctionEmbedding
                            mat2.append(c.embedding)
                            name_list.append(c.name)

                        if n_candidates:
                            if len(mat2) > n_candidates:
                                mat2 = random.sample(mat2, n_candidates)
                                name_list = random.sample(name_list, n_candidates)
                    else:
                        mat2.append(itself["embedding"])
                        name_list.append(itself["name"])

                        assert (
                            n_candidates > 0
                        ), "If you choose pool mode, you must specify the number of candidates"
                        for i in range(n_candidates):
                            arch, opt = random.choice(list(candidate_pool.keys()))
                            while (
                                arch == function_body["arch"]
                                or opt == function_body["opt"]
                            ):
                                arch, opt = random.choice(list(candidate_pool.keys()))
                            selected = random.choice(candidate_pool[(arch, opt)])
                            name_list.append(selected.name)
                            mat2.append(selected.embedding)

                    mat2 = np.vstack(mat2)
                    length.append(len(mat2))

                    # mm = get_cos_similar_multi(query_embedding, mat2)
                    # mm = similarity_score(query_embedding, mat2)
                    mm, ref = get_pearson_score(query_embedding, mat2)
                    rank_list = sorted(
                        zip(mm.reshape(-1), name_list), key=lambda x: x[0], reverse=True
                    )[: self.config.topK]
                    for k in range(1, max_k + 1):
                        is_correct = self.judge(name, [x[1] for x in rank_list[:k]])
                        record_total[k][0] += int(is_correct)
                        record_total[k][1] += 1

            success_result = True
            for k in range(1, max_k + 1):
                if record_total[k][1] == 0:
                    success_result = False
                    break
            if not success_result:
                continue

            for k in range(1, max_k + 1):
                recall[k].append(record_total[k][0] / record_total[k][1])
            # return

        recall_avg = []
        for k in range(1, max_k + 1):
            recall_avg.append(np.mean(recall[k]))

        pbar.close()
        # print("recall_avg", recall_avg, "\n")
        return recall_avg

    def test_recall_K_asm(
        self,
        dataset: dict,
        graph_path: str,
        max_k: int = 10,
        n_candidates: int = 100,
        cache_path: str = "",
    ):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k

        if cache_path:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    dataset = pickle.load(f)
                    f.close()
            else:
                graph, _ = dgl.load_graphs(graph_path)
                dataset = self.merge_dgl_dict(dataset, graph)
                with open(cache_path, "wb") as f:
                    pickle.dump(dataset, f)
                    f.close()
        else:
            graph, _ = dgl.load_graphs(graph_path)
            dataset = self.merge_dgl_dict(dataset, graph)

        pbar = tqdm(total=self.get_dataset_function_num(dataset))

        length = []
        for binary in dataset["data"]:
            record_total = {x: [0, 0] for x in range(1, max_k + 1)}
            print("Generating Function Pool for {}".format(binary))
            candidate_pool, candidate_name_list = self.get_function_file_set(
                dataset=dataset, binary_name=binary
            )
            # return
            function_list1 = dataset["data"][binary].keys()
            for function_name in function_list1:
                for function_body in dataset["data"][binary][function_name]:
                    pbar.update()

                    if len(dataset["data"][binary][function_name]) < 2:
                        continue

                    arch, opt = function_body["arch"], function_body["opt"]

                    if (arch, opt) not in candidate_pool:
                        continue

                    name, query_embedding = function_body["name"], FunctionEmbedding(
                        name=function_body["name"], embedding=function_body["embedding"]
                    )
                    query_embedding: FunctionEmbedding
                    query_embedding = query_embedding.embedding

                    mat2 = []
                    name_list = []

                    itself = random.choice(dataset["data"][binary][function_name])
                    random_count = 0
                    while itself["opt"] == opt:
                        itself = random.choice(dataset["data"][binary][function_name])
                        random_count += 1
                        if random_count > 100:
                            break

                    if random_count > 100:
                        continue

                    # mat2.append(function_body['embedding'])

                    assert (
                        n_candidates > 0
                    ), "If you choose pool mode, you must specify the number of candidates"
                    for i in range(n_candidates):
                        arch, opt = random.choice(list(candidate_pool.keys()))
                        while opt == function_body["opt"]:
                            arch, opt = random.choice(list(candidate_pool.keys()))
                        selected = random.choice(candidate_pool[(arch, opt)])
                        name_list.append(selected.name)
                        mat2.append(selected.embedding)

                    mat2.append(itself["embedding"])

                    mat2 = np.vstack(mat2)
                    length.append(len(mat2))

                    # mm = get_cos_similar_multi(query_embedding, mat2)
                    # mm = similarity_score(query_embedding, mat2)
                    # NaN output corrupt the result, change the seq to mitigate
                    mm, ref = get_pearson_score(query_embedding, mat2)
                    rank_list = sorted(
                        zip(mm.reshape(-1), name_list + [function_name]),
                        key=lambda x: x[0],
                        reverse=True,
                    )[: self.config.topK]
                    for k in range(1, max_k + 1):
                        is_correct = self.judge(name, [x[1] for x in rank_list[:k]])
                        record_total[k][0] += int(is_correct)
                        record_total[k][1] += 1

            success_result = True
            for k in range(1, max_k + 1):
                if record_total[k][1] == 0:
                    success_result = False
                    break
            if not success_result:
                continue

            for k in range(1, max_k + 1):
                recall[k].append(record_total[k][0] / record_total[k][1])
            # return

        recall_avg = []
        for k in range(1, max_k + 1):
            recall_avg.append(np.mean(recall[k]))

        pbar.close()
        # print("recall_avg", recall_avg, "\n")
        return recall_avg

    def test_recall_K_x86(
        self,
        dataset: dict,
        graph_path: str,
        max_k: int = 10,
        n_candidates: int = 100,
        cache_path: str = "",
    ):
        recall = {x: [] for x in range(1, max_k + 1)}
        self.config.topK = max_k

        if cache_path:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    dataset = pickle.load(f)
                    f.close()
            else:
                graph, _ = dgl.load_graphs(graph_path)
                dataset = self.merge_dgl_dict(dataset, graph)
                with open(cache_path, "wb") as f:
                    pickle.dump(dataset, f)
                    f.close()
        else:
            graph, _ = dgl.load_graphs(graph_path)
            dataset = self.merge_dgl_dict(dataset, graph)

        pbar = tqdm(total=self.get_dataset_function_num(dataset))

        length = []
        for binary in dataset["data"]:
            record_total = {x: [0, 0] for x in range(1, max_k + 1)}
            print("Generating Function Pool for {}".format(binary))
            candidate_pool, candidate_name_list = self.get_function_file_set(
                dataset=dataset, binary_name=binary
            )
            # return
            function_list1 = dataset["data"][binary].keys()
            for function_name in function_list1:
                for function_body in dataset["data"][binary][function_name]:
                    pbar.update()

                    if len(dataset["data"][binary][function_name]) < 2:
                        continue

                    arch, opt = function_body["arch"], function_body["opt"]

                    if (arch, opt) not in candidate_pool:
                        continue

                    name, query_embedding = function_body["name"], FunctionEmbedding(
                        name=function_body["name"], embedding=function_body["embedding"]
                    )
                    query_embedding: FunctionEmbedding
                    query_embedding = query_embedding.embedding

                    itself = random.choice(dataset["data"][binary][function_name])
                    random_count = 0
                    while itself["opt"] == opt:
                        itself = random.choice(dataset["data"][binary][function_name])
                        random_count += 1
                        if random_count > 100:
                            break

                    if random_count > 100:
                        continue
                    mat2 = [itself["embedding"]]
                    name_list = [function_name]
                    # mat2.append(function_body['embedding'])

                    assert (
                        n_candidates > 0
                    ), "If you choose pool mode, you must specify the number of candidates"
                    for i in range(n_candidates):
                        arch, opt = random.choice(list(candidate_pool.keys()))
                        while opt == function_body["opt"]:
                            arch, opt = random.choice(list(candidate_pool.keys()))
                        selected = random.choice(candidate_pool[(arch, opt)])
                        name_list.append(selected.name)
                        mat2.append(selected.embedding)

                    mat2.append(itself["embedding"])

                    mat2 = np.vstack(mat2)
                    length.append(len(mat2))

                    # mm = get_cos_similar_multi(query_embedding, mat2)
                    # mm = similarity_score(query_embedding, mat2)
                    mm, ref = get_pearson_score(query_embedding, mat2)
                    rank_list = sorted(
                        zip(mm.reshape(-1), name_list), key=lambda x: x[0], reverse=True
                    )[: self.config.topK]
                    for k in range(1, max_k + 1):
                        is_correct = self.judge(name, [x[1] for x in rank_list[:k]])
                        record_total[k][0] += int(is_correct)
                        record_total[k][1] += 1

            success_result = True
            for k in range(1, max_k + 1):
                if record_total[k][1] == 0:
                    success_result = False
                    break
            if not success_result:
                continue

            for k in range(1, max_k + 1):
                recall[k].append(record_total[k][0] / record_total[k][1])
            # return

        recall_avg = []
        for k in range(1, max_k + 1):
            recall_avg.append(np.mean(recall[k]))

        pbar.close()
        # print("recall_avg", recall_avg, "\n")
        return recall_avg

    # @profile
    def get_function_file_set(
        self, dataset: dict, binary_name
    ) -> Tuple[Dict[tuple, List[FunctionEmbedding]], Dict[tuple, List[str]]]:
        candidate_pool: Dict[tuple, List[FunctionEmbedding]] = {}
        candidate_name_pool: Dict[tuple, List[str]] = {}
        count = 0
        for function_name in dataset["data"][binary_name]:
            for function_body in dataset["data"][binary_name][function_name]:
                arch, opt = function_body["arch"], function_body["opt"]
                if (arch, opt) not in candidate_pool:
                    candidate_pool[(arch, opt)] = []
                    candidate_name_pool[(arch, opt)] = []

                name, embedding = function_body["name"], FunctionEmbedding(
                    name=function_body["name"], embedding=function_body["embedding"]
                )
                candidate_pool[(arch, opt)].append(embedding)
                candidate_name_pool[(arch, opt)].append(name)
                count += 1
                # if count >= 5:
                #     return candidate_pool, candidate_name_pool

        return candidate_pool, candidate_name_pool

    def AUC(self, dataset: dict, graphs: List[dgl.DGLGraph]):

        scores = []
        labels = []

        dataset = self.merge_dgl_dict(dataset, graphs)

        for binary_name in dataset["data"]:
            for function_name in dataset["data"][binary_name]:
                if len(dataset["data"][binary_name][function_name]) < 2:
                    continue
                for i in range(len(dataset["data"][binary_name][function_name])):
                    this_embedding = dataset["data"][binary_name][function_name][i][
                        "embedding"
                    ]

                    candidates = dataset["data"][binary_name][function_name].copy()
                    del candidates[i]
                    same = random.choice(candidates)
                    diff = random.choice(
                        self.get_different_function_sample(
                            dataset, binary_name, function_name
                        )
                    )
                    same_embedding = same["embedding"]
                    diff_embedding = diff["embedding"]

                    same_score = pearson_score(this_embedding, same_embedding)
                    diff_score = pearson_score(this_embedding, diff_embedding)

                    scores.extend([same_score, diff_score])
                    labels.extend([1, 0])

        roc_score = roc_auc_score(y_true=labels, y_score=scores)
        print(roc_score)
        return roc_score

    def get_different_function_sample(
        self, dataset: dict, binary_name: str, function_name: str
    ):
        selected_binary_name = random.choice(list(dataset["data"].keys()))
        selected_function_name = random.choice(
            list(dataset["data"][selected_binary_name].keys())
        )
        while (
            selected_binary_name == binary_name
            and selected_function_name == function_name
        ):
            selected_binary_name = random.choice(list(dataset["data"].keys()))
            selected_function_name = random.choice(
                list(dataset["data"][selected_binary_name].keys())
            )
        return random.choice(
            dataset["data"][selected_binary_name][selected_function_name]
        )

    def get_dataset_function_num(self, dataset: dict):
        num = 0
        for binary_name in dataset["data"]:
            for function_name in dataset["data"][binary_name]:
                num += len(dataset["data"][binary_name][function_name])
        return num


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        f.close()
    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        f.close()


def load_model(path, use_cuda=True):
    model_config = ModelConfig()
    model_config.model_path = path
    model_config.dataset_path = ""
    model_config.feature_length = 151
    model_config.max_length = 1000
    model_config.cuda = use_cuda
    model_config.topK = 50
    model = InferenceModel(model_config)
    return model


def evaluate_dataset_1(pretrained_model, dataset_folder, use_cuda=True):

    model = load_model(pretrained_model, use_cuda)

    graph_path = os.path.join(dataset_folder, "dataset_1", "dgl_graphs.dgl")

    dataset = load_pickle(
        os.path.join(dataset_folder, "dataset_1", "index_test_data_{}.pkl".format(5))
    )

    res = model.test_recall_K(
        dataset, graph_path, max_k=50, mode="pool", n_candidates=100, cache_path=""
    )

    return res


def evaluate_dataset_1_asm_x86(
    pretained_model_asm, pretrained_model_x86, dataset_folder
):

    model_asm = load_model(pretained_model_asm, use_cuda=True)

    graph_path = os.path.join(dataset_folder, "dataset_1_asm", "dgl_graphs.dgl")

    dataset = load_pickle(
        os.path.join(
            dataset_folder, "dataset_1_asm", "index_test_data_{}.pkl".format(5)
        )
    )

    res_asm = model_asm.test_recall_K_asm(
        dataset, graph_path, max_k=50, n_candidates=100, cache_path=""
    )

    model_x86 = load_model(pretrained_model_x86)

    graph_path = os.path.join(dataset_folder, "dataset_1_x86", "dgl_graphs.dgl")

    dataset = load_pickle(
        os.path.join(
            dataset_folder, "dataset_1_x86", "index_test_data_{}.pkl".format(5)
        )
    )

    res_x86 = model_x86.test_recall_K_x86(
        dataset, graph_path, max_k=50, n_candidates=100, cache_path=""
    )

    return res_asm, res_x86


def evaluate_dataset_2(pretrained_model, dataset_folder, use_cuda=True):

    model = load_model(pretrained_model, use_cuda)

    total_res = []
    graph_strip_path = os.path.join(dataset_folder, "dataset_2", "dgl_graphs.dgl")

    graph_strip, _ = dgl.load_graphs(graph_strip_path)
    for i in range(1, 6):
        print("Running Fold: ", i)
        dataset_origin = load_pickle(
            os.path.join(
                dataset_folder, "dataset_2", "index_test_data_{}.pkl".format(i)
            )
        )
        res, ref = model.test_strip_recall_K(
            dataset_origin=dataset_origin,
            dataset_strip=dataset_origin,
            graph_strip=graph_strip,
            max_k=50,
            n_candidates=100,
        )
        total_res.append(res)

    total_res = np.array(total_res)
    mean_res = np.mean(total_res, axis=0)
    return mean_res


def evaluate_dataset_2_strip(pretrained_model, dataset_folder, use_cuda=True):
    model = load_model(pretrained_model, use_cuda)

    total_res = []
    graph_strip_path = os.path.join(
        dataset_folder, "dataset_2_decompressed_stripped", "dgl_graphs.dgl"
    )
    graph_strip, _ = dgl.load_graphs(graph_strip_path)
    for i in range(1, 6):
        print("Running Fold: ", i)

        dataset_strip = load_pickle(
            os.path.join(
                dataset_folder,
                "dataset_2_decompressed_stripped",
                "index_test_data_{}.pkl".format(i),
            )
        )
        dataset_origin = load_pickle(
            os.path.join(
                dataset_folder,
                "dataset_2_decompressed_nostrip",
                "index_test_data_{}.pkl".format(i),
            )
        )

        res, _ = model.test_strip_recall_K(
            dataset_origin=dataset_origin,
            dataset_strip=dataset_strip,
            graph_strip=graph_strip,
            max_k=50,
            n_candidates=100,
        )
        total_res.append(res)

    total_res = np.array(total_res)
    mean_res = np.mean(total_res, axis=0)
    return mean_res


def evaluate_dataset_openplc(
    pretrained_model, openplc_dataset_folder, index_test_data_path, top_k, use_cuda=True
):
    model = load_model(pretrained_model, use_cuda)
    model.config.topK = top_k
    graph_path = os.path.join(openplc_dataset_folder, "dgl_graphs.dgl")
    dataset = load_pickle(index_test_data_path)
    res = model.test_recall_K(
        dataset, graph_path, max_k=50, mode="pool", n_candidates=100, cache_path=""
    )
    return res


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--pretrained_model",
        required=True,
        help="Path to the pretrained model folder",
    )
    ap.add_argument(
        "-d", "--graph_dataset", required=True, help="Path to the graph dataset folder"
    )
    ap.add_argument(
        "-c",
        "--cpu",
        action="store_true",
        help="Enforce using CPU or not, adding this flag will enforce using CPU",
    )
    args = ap.parse_args()
    return args


def plot_recall_k(ax: plt.Axes, label, recall, label2=None, recall2=None):

    if isinstance(recall, list):
        recall = np.array(recall)

    if isinstance(recall2, list):
        recall2 = np.array(recall2)

    colors = ["#1e1a22", "#67a4ba", "#4e4c72", "#b94e5e", "#d389a1"]
    recall_index = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    recall_index = np.array(recall_index)
    recall_value = recall[recall_index - 1]
    if recall2 is not None:
        recall_value2 = recall2[recall_index - 1]

    ax.plot(
        recall_index, recall_value, label=label, marker="o", color=random.choice(colors)
    )
    if recall2 is not None:
        ax.plot(
            recall_index,
            recall_value2,
            label=label2,
            marker="o",
            color=random.choice(colors),
        )
    ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 51)
    ax.set_xticks(recall_index)
    ax.grid()


def plot_results(
    dataset_1_result,
    dataset_2_result,
    dataset_2_strip_result,
    dataset_1_asm_result,
    dataset_1_x86_result,
):
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_recall_k(ax[0][0], "Dataset-1", dataset_1_result)
    plot_recall_k(ax[0][1], "Dataset-2", dataset_2_result)
    plot_recall_k(ax[1][0], "Dataset-2-strip", dataset_2_strip_result)
    plot_recall_k(
        ax[1][1],
        "Dataset-1-asm",
        dataset_1_asm_result,
        "Dataset-1-x86",
        dataset_1_x86_result,
    )
    fig.suptitle("Recall@K")
    plt.tight_layout()
    plt.savefig("./recall@k.pdf")


if __name__ == "__main__":
    args = parse_args()
    graph_dataset = args.graph_dataset
    pretrained_model = args.pretrained_model
    use_cuda = not args.cpu

    dataset_1_model = os.path.join(pretrained_model, "dataset_1.ckpt")
    dataset_1_result = evaluate_dataset_1(dataset_1_model, graph_dataset, use_cuda)

    dataset_2_model = os.path.join(pretrained_model, "dataset_2_all.ckpt")
    dataset_2_result = evaluate_dataset_2(dataset_2_model, graph_dataset, use_cuda)

    dataset_2_strip_result = evaluate_dataset_2_strip(
        dataset_2_model, graph_dataset, use_cuda
    )

    dataset_1_asm_model = os.path.join(pretrained_model, "dataset_1_asm.ckpt")
    dataset_1_x86_model = os.path.join(pretrained_model, "dataset_1_x86.ckpt")
    dataset_1_asm_result, dataset_1_x86_result = evaluate_dataset_1_asm_x86(
        dataset_1_asm_model, dataset_1_x86_model, graph_dataset, use_cuda
    )
    pickle_to_save = {
        "dataset_1": dataset_1_result,
        "dataset_2": dataset_2_result,
        "dataset_2_strip": dataset_2_strip_result,
        "dataset_1_asm": dataset_1_asm_result,
        "dataset_1_x86": dataset_1_x86_result,
    }
    save_pickle(pickle_to_save, "recall@k.pkl")

    plot_results(
        dataset_1_result,
        dataset_2_result,
        dataset_2_strip_result,
        dataset_1_asm_result,
        dataset_1_x86_result,
    )
