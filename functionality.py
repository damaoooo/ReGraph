import argparse
import os
import subprocess
import shutil
import dgl
import numpy as np
import pandas as pd
import torch
from PreProcessor.GraphConverter import Converter
from inference import load_model, get_pearson_score
from tqdm import tqdm

JOERN_PATH="/home/damaoooo/Downloads/joern-cli"

def clean_ext(folder, ext):
    for file in os.listdir(folder):
        if file.endswith(ext):
            os.remove(os.path.join(folder, file))


def clean_dir(folder):
    clean_ext(folder, ".dsm")
    clean_ext(folder, ".config.json")
    clean_ext(folder, ".bc")
    clean_ext(folder, ".c")
    clean_ext(folder, ".ll")
    clean_ext(folder, ".cpg")
    clean_ext(folder, ".re")

    # if folder "workspace" or "out" exists, remove it
    if os.path.exists(folder + "/workspace"):
        shutil.rmtree(folder + "/workspace")
    if os.path.exists(folder + "/out"):
        shutil.rmtree(folder + "/workspace")


def lift_and_reoptimize(file_path, timeout=3600):
    current_env = os.environ.copy()
    current_env["LD_LIBRARY_PATH"] = "/usr/local/lib"

    try:
        m = subprocess.run(
            [
                "retdec-decompiler",
                file_path,
                "-s",
                "-k",
                "--backend-keep-library-funcs",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        m = subprocess.run(
            ["llvm-dis", file_path + ".bc", "-o", file_path + ".ll"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        m = subprocess.run(
            [
                "clang",
                "-m32",
                "-O3",
                "-c",
                file_path + ".ll",
                "-fno-inline-functions",
                "-o",
                file_path + ".re",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=current_env,
        )
        m = subprocess.run(
            [
                "retdec-decompiler",
                file_path + ".re",
                "-s",
                "-k",
                "--backend-keep-library-funcs",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print("Timeout")
        return False
    return True


def generate_cpg(file_path, timeout=3600):
    joern_path = JOERN_PATH
    current_env = os.environ.copy()
    current_env["LD_LIBRARY_PATH"] = "/usr/local/lib"
    file_path, file_name = os.path.split(file_path)
    c2cpg_path = os.path.join(joern_path, "c2cpg.sh")
    joern_export_path = os.path.join(joern_path, "joern-export")
    try:
        m = subprocess.run(
            [c2cpg_path, file_name + ".re.c", "-o", file_name + ".cpg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=current_env,
        )
        m = subprocess.run(
            [
                joern_export_path,
                file_name + ".cpg",
                "-o",
                "c_dot_" + file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=current_env,
        )
    except subprocess.TimeoutExpired:
        print("Timeout")
        return False
    return True


def c_dot_to_dgl(c_dot_path, convertor: Converter, binary_name, arch, opt):
    pool = []
    for file in os.listdir(c_dot_path):
        if not file.endswith(".dot"):
            continue
        ret = convertor.convert_file(
            os.path.join(c_dot_path, file), binary_name=binary_name, arch=arch, opt=opt
        )
        if ret is None:
            continue
        pool.append(ret)

    dgl_pool = []
    index_pool = []
    for function_body in pool:

        function_name = function_body["name"]
        binary_name = function_body["binary"]

        graph = dgl.graph((function_body["adj"][0], function_body["adj"][1]))
        graph = dgl.to_bidirected(graph)
        graph: dgl.DGLGraph = dgl.add_self_loop(graph)
        try:
            graph.ndata["feat"] = torch.tensor(
                function_body["feature"], dtype=torch.float
            )
        except dgl._ffi.base.DGLError:
            print(
                f"Bad Function detected from {binary_name} - {function_name}, need to more sanitize"
            )
            continue
        del function_body["adj"]
        del function_body["feature"]

        function_body["index"] = len(dgl_pool)

        dgl_pool.append(graph)
        index_pool.append(function_body)

    return dgl_pool, index_pool


def convert_to_embedding(dgl_pool, index_file, model_path):
    with torch.no_grad():
        model = load_model(model_path, use_cuda=False)
        
        # 设置批处理大小
        batch_size = 256
        total_samples = len(index_file)
        
        for idx in range(len(dgl_pool)):
            data = dgl_pool[idx]
            padding = model.max_length - data.num_nodes()
            data = dgl.add_nodes(data, padding)
            data = dgl.add_self_loop(data)
            dgl_pool[idx] = data
        
        # 创建进度条
        pbar = tqdm(total=total_samples)
        pbar.set_description("Converting to embeddings")
        
        # 预分配结果列表
        results = [None] * total_samples
        
        # 批量处理
        for i in range(0, total_samples, batch_size):
            batch_indices = range(i, min(i + batch_size, total_samples))
            batch_graphs = []
            
            # 准备批处理的图
            for idx in batch_indices:
                graph = dgl_pool[index_file[idx]["index"]]
                batch_graphs.append(graph)
            
            # 批处理
            if len(batch_graphs) > 0:
                batched_graphs = dgl.batch(batch_graphs)
                embeddings = model.single_dgl_to_embedding(batched_graphs)
                
                # 将结果存储到预分配的列表中
                for idx, embedding in zip(batch_indices, np.split(embeddings, len(batch_graphs))):
                    results[idx] = embedding
            
            # 更新进度条
            pbar.update(len(batch_indices))
        
        # 将结果写回到index_file中
        for idx, embedding in enumerate(results):
            index_file[idx]["embedding"] = embedding
        
        pbar.close()
        return index_file


def compute_similarity(index_file_1, index_file_2, topK=10):
    result = {}
    for index_1 in range(len(index_file_1)):
        ref_embedding = index_file_1[index_1]["embedding"]
        candidate_embedding = []
        name_list = []
        for index_2 in range(len(index_file_2)):
            candidate_embedding.append(index_file_2[index_2]["embedding"])
            name_list.append(index_file_2[index_2]["name"])
        candidate_embedding = np.vstack(candidate_embedding)

        mm, ref = get_pearson_score(ref_embedding, candidate_embedding)
        rank_list = sorted(
            zip(name_list, mm.reshape(-1)), key=lambda x: x[1], reverse=True
        )[:topK]

        name_1 = index_file_1[index_1]["name"]
        result[name_1] = rank_list
    return result


def output_result(data: dict):

    rows = []
    print(data)
    # 遍历字典中的每个键值对
    for func, values in data.items():
        # 创建包含函数名的行
        row = [func]
        for name, score in values:
            row.extend([score, name])
        rows.append(row)

    # 找到最长的行长度，确保所有行具有相同的长度
    max_length = max(len(row) for row in rows)

    # 将所有行填充到相同的长度
    for row in rows:
        while len(row) < max_length:
            row.append(None)

    # 创建DataFrame
    columns = ["Function"] + [
        f"{t}{i}" for i in range(1, (max_length // 2) + 1) for t in ("Score", "Name")
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df


def inference(input_1, input_2, model_file, op_file, top_k):

    # Make temp folder and move input files to temp folder
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    else:
        shutil.rmtree(os.path.abspath(temp_folder))
        os.mkdir(temp_folder)
    shutil.copy(input_1, temp_folder)
    shutil.copy(input_2, temp_folder)

    temp_full_path = os.path.join(os.getcwd(), temp_folder)

    input_1 = os.path.join(temp_folder, os.path.split(input_1)[1])
    input_2 = os.path.join(temp_folder, os.path.split(input_2)[1])

    input_1 = os.path.abspath(input_1)
    input_2 = os.path.abspath(input_2)

    model_file = os.path.abspath(model_file)
    op_file = os.path.abspath(op_file)

    previoud_dir = os.getcwd()
    os.chdir(temp_folder)

    lift_and_reoptimize(input_1)
    lift_and_reoptimize(input_2)

    generate_cpg(input_1)
    generate_cpg(input_2)

    convertor = Converter(op_file=op_file)

    # get input_1 c_dot path
    def conver_to_c_dot_path(path):
        return os.path.join(os.path.split(path)[0], "c_dot_" + os.path.split(path)[1])

    dgl_1, index_1 = c_dot_to_dgl(
        conver_to_c_dot_path(input_1), convertor, "binary", "arch", "opt"
    )
    dgl_2, index_2 = c_dot_to_dgl(
        conver_to_c_dot_path(input_2), convertor, "binary", "arch", "opt"
    )

    index_1 = convert_to_embedding(dgl_1, index_1, model_file)
    index_2 = convert_to_embedding(dgl_2, index_2, model_file)

    result_data = compute_similarity(index_1, index_2, top_k)
    data_frame = output_result(result_data)

    # clean temp folder
    clean_dir(".")
    os.chdir(previoud_dir)

    # remove temp folder
    shutil.rmtree(temp_full_path)
    input_1_name = os.path.split(input_1)[1]
    input_2_name = os.path.split(input_2)[1]
    save_file_name = f"{input_1_name}_vs_{input_2_name}_result.xlsx"
    data_frame.to_excel(save_file_name, index=False)
    return data_frame, result_data


def print_result(data: dict):
    for function_name in data:
        print(
            f"Function: {function_name} | Top K Similar Functions:",
            [x[0] for x in data[function_name]],
        )


# def get_args():
#     parser = argparse.ArgumentParser(description="Functionality Test")
#     parser.add_argument("--input1", type=str, help="Input file 1", required=True)
#     parser.add_argument("--input2", type=str, help="Input file 2", required=True)
#     parser.add_argument("--model", type=str, help="Model file", required=True)
#     parser.add_argument("--op_file", type=str, help="Operator file", required=True)
#     parser.add_argument("-k", type=int, help="Top k", default=10)
#     args = parser.parse_args()
#     return args

class Args:
    def __init__(self):
        self.input1 = ""
        self.input2 = ""
        self.model = ""
        self.op_file = ""
        self.k = 10

if __name__ == "__main__":
    # args = get_args()
    
    args = Args()
    args.input1 = "./graph_dataset/openplc/g++-O0/Res0_g++-O0.o"
    args.input2 = "./graph_dataset/openplc/arm-linux-gnueabi-g++-O3/Res0_arm-linux-gnueabi-g++-O3.o"
    args.model = "./pretrained/openplc.ckpt"
    args.op_file = "./graph_dataset/openplc/op_file.pkl"
    args.k = 10
    
    data_frame, result_data = inference(
        args.input1, args.input2, args.model, args.op_file, args.k
    )
    print_result(result_data)
