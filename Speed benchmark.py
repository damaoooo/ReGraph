from model import MyModel
import dgl
import time
import torch
import argparse

parser = argparse.ArgumentParser(description="Speed benchmark")
parser.add_argument("-N", "--max_node", type=int, default=500, help="max node number")
parser.add_argument(
    "-B", "--batch_size", type=int, default=100, help="how many graphs in a test"
)
parser.add_argument("-T", "--test_times", type=int, default=100, help="test times")
parser.add_argument("--cpu_only", action="store_true", help="only test CPU")
args = parser.parse_args()

max_node = args.max_node
batch_size = args.batch_size

test_times = args.test_times
if not args.cpu_only:
    dummy_graph = dgl.graph(
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        num_nodes=max_node,
    )
    dummy_graph = dgl.add_self_loop(dummy_graph)
    dummy_graph.ndata["feat"] = torch.randn(max_node, 151)
    dummy_graph = dgl.batch([dummy_graph] * batch_size)
    dummy_graph = dummy_graph.to("cuda")

    print("CUDA FP32 testing... ")
    model = MyModel(151, 128, 64, 6, 0.3, 0.2, max_node)
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for i in range(test_times):
            s = model(dummy_graph)
        end_time = time.time()
        print(
            "CUDA FP32 time: ",
            round((end_time - start_time) / test_times, 4),
            "s / {} functions".format(batch_size),
        )

    print("CUDA FP16 testing... ")
    model = MyModel(151, 128, 64, 6, 0.3, 0.2, 500)
    model = model.cuda()
    model.eval()

    dummy_graph = dgl.graph(
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        num_nodes=max_node,
    )
    dummy_graph = dgl.add_self_loop(dummy_graph)
    dummy_graph.ndata["feat"] = torch.randn(max_node, 151)
    dummy_graph = dgl.batch([dummy_graph] * batch_size)
    dummy_graph = dummy_graph.to("cuda")

    scaler = torch.cuda.amp.GradScaler()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            start_time = time.time()
            for i in range(test_times):
                s = model(dummy_graph)
            end_time = time.time()
            print(
                "CUDA FP16 time: ",
                round((end_time - start_time) / test_times, 4),
                "s / {} functions".format(batch_size),
            )


dummy_graph = dgl.graph(
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_nodes=max_node
)
dummy_graph = dgl.add_self_loop(dummy_graph)
dummy_graph.ndata["feat"] = torch.randn(max_node, 151)
dummy_graph = dgl.batch([dummy_graph] * batch_size)
dummy_graph = dummy_graph.to("cpu")

print("CPU testing... ")
model = MyModel(151, 128, 64, 6, 0.3, 0.2, 500)
model.eval()

with torch.no_grad():
    start_time = time.time()
    for i in range(test_times):
        s = model(dummy_graph)
    end_time = time.time()
    print(
        "CPU time: ",
        round((end_time - start_time) / test_times, 4),
        "s / {} functions".format(batch_size),
    )
