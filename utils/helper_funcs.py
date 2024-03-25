import numpy as np
from torch_geometric.data import Data, Batch
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
import os
import subprocess
import time


def split_communities(communities, n_train, n_val=0, already_train__test=False, dataset='amazon', time=0, mapping=None):
    r"""Randomly split all communities into train set, validation set, and test set"""
    if already_train__test:
        train_val_comms = []
        test_comms = []
        train_path = f'./dataset/{dataset}/time_{time}/train_set.txt'
        test_path = f'./dataset/{dataset}/time_{time}/test_set.txt'
        with open(train_path, 'r') as file:
            for line in file:
                comm = list(map(int, line.split()))
                comm = [mapping[node] for node in comm]
                train_val_comms.append(comm)
        with open(test_path, 'r') as file:
            for line in file:
                comm = list(map(int, line.split()))
                comm = [mapping[node] for node in comm]
                test_comms.append(comm)
        # Extract validation set from train set
        idxes = list(range(len(train_val_comms)))
        np.random.shuffle(idxes)
        val_comms = [train_val_comms[idx] for idx in idxes[:n_val]]
        train_comms = [train_val_comms[idx] for idx in idxes[n_val:]]
    else:
        idxes = list(range(len(communities)))
        np.random.shuffle(idxes)

        train_comms = [communities[idx] for idx in idxes[:n_train]]
        val_comms = [communities[idx] for idx in idxes[n_train:n_train + n_val]]
        test_comms = [communities[idx] for idx in idxes[n_train + n_val:]]

    return train_comms, val_comms, test_comms


def drop_nodes(graph_data, aug_ratio=0.15):
    r"""Contrastive model corruption: dropping nodes"""
    node_num, edge_num = graph_data.x.size(0), graph_data.edge_index.size(1)

    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_non_drop = idx_perm[drop_num:]
    idx_non_drop.sort()

    idx_dict = {idx_non_drop[n]: n for n in list(range(idx_non_drop.shape[0]))}

    device = graph_data.edge_index.device

    edge_index = graph_data.edge_index.detach().cpu().numpy()

    # re-label edges
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]

    try:
        # create a corrupted subgraph
        new_edge_index = torch.tensor(edge_index).transpose_(0, 1).to(device)
        new_x = graph_data.x[idx_non_drop]
        new_graph_data = Data(x=new_x, edge_index=new_edge_index)
    except:
        new_graph_data = graph_data
    return new_graph_data


def prepare_locator_train_data(node_list, data: Data, max_size=25, num_hop=2):
    r"""Generate batch data for Community Locator training. For each node,
    extract its ego-net and generate a sub-ego-net"""
    batch, subg_batch = [], []

    num_nodes = data.x.size(0)

    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(node_idx=node, num_hops=num_hop, edge_index=data.edge_index,
                                           num_nodes=num_nodes)

        if len(node_set) > max_size:
            node_set = node_set[torch.randperm(node_set.shape[0])][:max_size]
            node_set = torch.unique(torch.cat([torch.LongTensor([node]), torch.flatten(node_set)]))

        node_list = node_set.detach().cpu().numpy().tolist()
        seed_idx = node_list.index(node)

        if seed_idx != 0:
            node_list[seed_idx], node_list[0] = node_list[0], node_list[seed_idx]

        # Hint: important!!!
        #  We must ensure all the first node is the centric node
        assert node_list[0] == node
        # print(node, node_list)

        edge_index, _ = subgraph(node_list, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        node_x = data.x[node_list]  # node features
        g_data = Data(x=node_x, edge_index=edge_index)
        batch.append(g_data)

        # apply **Drop-Node** to obtain its subgraph
        corrupt_data = drop_nodes(g_data)
        subg_batch.append(corrupt_data)
        # print(g_data, corrupt_data)

    batch = Batch().from_data_list(batch)
    subg_batch = Batch().from_data_list(subg_batch)

    return batch, subg_batch


def new_prepare_locator_train_data(node_list, data: Data, max_size=25, num_hop=2):
    r"""Generate batch data for Community Locator training. For each node,
    extract its ego-net and generate a sub-ego-net"""
    candidate_communities = []

    num_nodes = data.x.size(0)

    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(node_idx=node, num_hops=num_hop, edge_index=data.edge_index,
                                           num_nodes=num_nodes)

        if len(node_set) > max_size:
            node_set = node_set[torch.randperm(node_set.shape[0])][:max_size]
            node_set = torch.unique(torch.cat([torch.LongTensor([node]), torch.flatten(node_set)]))

        node_list = node_set.detach().cpu().numpy().tolist()
        seed_idx = node_list.index(node)

        if seed_idx != 0:
            node_list[seed_idx], node_list[0] = node_list[0], node_list[seed_idx]

        # Hint: important!!!
        #  We must ensure all the first node is the centric node
        assert node_list[0] == node
        # print(node, node_list)

        candidate_communities.append(node_list)

    return candidate_communities



def generate_ego_net(graph, start_node, k=1, max_size=15, choice="subgraph"):
    """Generate **k** ego-net"""
    q = [start_node]
    visited = [start_node]

    iteration = 0
    while True:
        if iteration >= k:
            break
        length = len(q)
        if length == 0 or len(visited) >= max_size:
            break

        for i in range(length):
            # Queue pop
            u = q[0]
            q = q[1:]

            for v in list(graph.neighbors(u)):
                if v not in visited:
                    q.append(v)
                    visited.append(v)
                if len(visited) >= max_size:
                    break
            if len(visited) >= max_size:
                break
        iteration += 1
    visited = sorted(visited)
    # print(visited)
    return visited if choice == "neighbors" else graph.subgraph(visited)


def generate_outer_boundary(graph, com_nodes, max_size=20):
    """For a given graph and a community `com_nodes`, generate its outer-boundary"""
    outer_nodes = []
    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)
    return outer_nodes if len(outer_nodes) <= max_size else outer_nodes[:max_size]


def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")


def count_folders_starting_with_time(path):
    # Initialize a counter for the folders starting with 't'
    count = 0

    # Get a list of all items in the specified path
    items = os.listdir(path)

    # Iterate through the items
    for item in items:
        # Check if the item is a directory and starts with 't'
        if os.path.isdir(os.path.join(path, item)) and item.startswith('time'):
            count += 1

    return count
