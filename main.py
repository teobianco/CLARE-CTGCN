import sys
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from Rewriter import CommRewriting
from Locator import CommMatching, gnn_embedding
from utils import split_communities, eval_scores, prepare_data, count_folders_starting_with_time
import os
from utils.helper_funcs import assign_free_gpus

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write2file(comms, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in comms])
        fh.write(content)


def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred


def parse_args(args):
    parser = argparse.ArgumentParser()
    # # CTGCN
    # parser = argparse.ArgumentParser(prog='CTGCN', description='K-core based Temporal Graph Convolutional Network')
    # parser.add_argument('--config', nargs=1, type=str, help='configuration file path', required=True)
    # parser.add_argument('--task', type=str, default='embedding', help='task name which is needed to run', required=True)
    # parser.add_argument('--method', type=str, default=None, help='graph embedding method, only used for embedding task')
    # General Config
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument("--device", dest="device", type=str, help="training device", default="cuda:0")
    parser.add_argument("--dataset", type=str, help="dataset", default="amazon")
    #   --in CLARE paper, we predict 1000 communities from 100 communities as a default setting
    parser.add_argument("--num_pred", type=int, help="pred size", default=1000)
    parser.add_argument("--num_train", type=int, help="pred size", default=90)
    parser.add_argument("--num_val", type=int, help="pred size", default=10)
    parser.add_argument("--already_train_test", type=bool, help="If the train and test communities are already defined",
                        default=True)
    parser.add_argument("--multiplier", type=float, help="multiplier", default=1.0)

    # Community Locator related
    #   --GNNEncoder Setting
    parser.add_argument("--gnn_type", type=str, help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--hidden_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=64)
    #   --Order Embedding Setting
    parser.add_argument("--margin", type=float, help="margin loss", default=0.6)
    #   --Generation
    parser.add_argument("--comm_max_size", type=int, help="Community max size", default=12)
    #   --Training
    parser.add_argument("--locator_lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--locator_epoch", type=int, default=30)
    parser.add_argument("--locator_batch_size", type=int, default=256)

    # Community Rewriter related
    parser.add_argument("--agent_lr", type=float, help="CommR learning rate", default=1e-3)
    #    -- for DBLP, the setting of n_eisode and n_epoch is a little picky
    parser.add_argument("--n_episode", type=int, help="number of episode", default=10)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=1000)
    parser.add_argument("--gamma", type=float, help="CommR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=10)
    parser.add_argument("--max_rewrite_step", type=int, help="", default=4)
    parser.add_argument("--commr_path", type=str, help="CommR path", default="")

    # Save log
    parser.add_argument("--writer_dir", type=str, help="Summary writer directory", default="")
    return parser.parse_args(args)


def parse_json_args(file_path):
    config_file = open(file_path)
    json_config = json.load(config_file)
    config_file.close()
    return json_config


def preprocessing_task(method, args):
    from preprocessing import preprocess
    assert method in ['GCN', 'GCN_TG', 'GAT', 'GAT_TG', 'SAGE', 'SAGE_TG', 'GIN', 'GIN_TG', 'PGNN', 'CGCN-C', 'GCRN', 'EvolveGCN', 'CTGCN-C']
    preprocess(method, args[method])


def embedding_task(method, args):
    print(args)

    # from baseline.dynAE import dyngem_embedding
    # from baseline.timers import timers_embedding
    # from train import gnn_embedding
    args['has_cuda'] = True if torch.cuda.is_available() else False

    if not args['has_cuda'] and 'use_cuda' in args and args['use_cuda']:
        # raise Exception('No CUDA devices is available, but you still try to use CUDA!')
        args['use_cuda'] = False
    if 'use_cuda' in args:
        args['has_cuda'] &= args['use_cuda']
    if not args['has_cuda']:  # Use CPU
        torch.set_num_threads(args['thread_num'])

    gnn_embedding(method, args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # print('args:', args)
    # config_dict = parse_json_args(args.config[0])
    seed_all(args.seed)
    #
    # # If run on GPU, we need to assign free GPUs
    # if args.device == "cuda:0":
    #     assign_free_gpus(max_gpus=1)
    #
    # if args.task == 'preprocessing':
    #     args_dict = config_dict[args.task]
    #     if args.method is None:
    #         raise AttributeError('Embedding method parameter is needed for the preprocessing task!')
    #     preprocessing_task(args.method, args_dict)
    # elif args.task == 'embedding':
    #     args_dict = config_dict[args.task]
    #     if args.method is None:
    #         raise AttributeError('Embedding method parameter is needed for the graph embedding task!')
    #     param_dict = args_dict[args.method]
    #     if not args['already_embedding']:
    #         embedding_task(args.method, param_dict)
    # else:
    #     raise AttributeError('Unsupported task!')

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    if not os.path.exists(f"ckpts/{args.dataset}"):
        os.mkdir(f"ckpts/{args.dataset}")

    f_list = []
    j_list = []
    nmi_list = []

    # args.writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.comm_max_size = 20 if args.dataset.startswith("lj") else 12

    time_len = count_folders_starting_with_time(f"./dataset/{args.dataset}/") 
    for time in range(time_len):

        args.writer_dir = f"ckpts/{args.dataset}/{time}"
        if not os.path.exists(args.writer_dir):
            os.mkdir(args.writer_dir)

        num_node, num_edge, num_community, graph_data, nx_graph, communities, mapping = prepare_data(args.dataset, time)
        train_comms, val_comms, test_comms = split_communities(communities, args.num_train, args.num_val,
                                                               args.already_train_test, args.dataset, time, mapping)

        ##########################################################
        ################### Step 2 Train Locator##################
        ##########################################################
        CommM_obj = CommMatching(args, graph_data, train_comms, val_comms, time, device=torch.device(args.device), mapping=mapping)
        # CommM_obj.train()
        pred_comms = CommM_obj.predict_community(nx_graph, args.comm_max_size)
        f1, jaccard, onmi = eval_scores(pred_comms, test_comms, train_comms, val_comms, tmp_print=True)
        metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
        write2file(pred_comms, args.writer_dir + "/CommM_" + metrics_string + '.txt')

        ##########################################################
        ################### Step 3 Train Rewriter#################
        ##########################################################
        cost_choice = "f1"  # or you can change to "jaccard"
        feat_mat = CommM_obj.generate_all_node_emb().detach().cpu().numpy()  # all nodes' embedding
        CommR_obj = CommRewriting(args, nx_graph, feat_mat, train_comms, val_comms, pred_comms, cost_choice)
        CommR_obj.train()
        rewrite_comms = CommR_obj.get_rewrite()
        # DA AGGIUNGERE CHE FARE CON LE RIPETIZIONI DEI NODI IN COMUNITà DIVERSE  --> LASCIARE COSì
        # CAPIRE SE ELIMINARE NODI CHE SONO NEL TRAIN SET  --> FATTO, PROVARE SE FUNZIONA
        f1, jaccard, onmi = eval_scores(rewrite_comms, test_comms, train_comms, val_comms, tmp_print=True)
        f_list.append(f1)
        j_list.append(jaccard)
        nmi_list.append(onmi)
        metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
        write2file(rewrite_comms, args.writer_dir + f"/CommR_{cost_choice}_" + metrics_string + '.txt')

    print(f'Mean F1: {np.mean(f_list)}')
    print(f'Std F1: {np.std(f_list)}')
    print(f'Mean Jaccard: {np.mean(j_list)}')
    print(f'Std Jaccard: {np.std(j_list)}')
    print(f'Mean ONMI: {np.mean(nmi_list)}')
    print(f'Std ONMI: {np.std(nmi_list)}')
    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
