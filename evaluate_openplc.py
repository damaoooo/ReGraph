from inference import evaluate_dataset_openplc
import os
import argparse


def evaluate(checkpoint_path: str, dataset_path: str, index_file_path: str, top_k: int):
    result = evaluate_dataset_openplc(
        checkpoint_path, dataset_path, index_file_path, top_k
    )
    for i in range(len(result)):
        print("Recall@{}: {}".format(i + 1, round(result[i], 4)))


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenPLC")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
    parser.add_argument("--index_file", type=str, help="Index file path", required=True)
    parser.add_argument("-k", type=int, help="Top k", default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args.checkpoint, args.dataset, args.index_file, args.k)
