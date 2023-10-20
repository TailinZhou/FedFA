
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated argumentss
    parser.add_argument('--E', type=int, default=5, help='number of rounds of local training')
    parser.add_argument('--r', type=int, default=200, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=100, help='number of total clients')
    parser.add_argument('--B', type=int, default=64, help='local batch size')#128
    parser.add_argument('--TB', type=int, default=1000, help="test batch size")
    parser.add_argument('--C', type=float, default=0.1, help='client samspling rate')
    
    # optimizer argumentss
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.99, help='learning rate decay per global round')
    parser.add_argument('--mu', type=float, default=0.1, help='proximal term constant for fedprox')
    parser.add_argument('--mu1', type=float, default=1, help='proximal term constant for moon')
    parser.add_argument('--alph', type=float, default=0.1, help='proximal term constant for feddyn')
    parser.add_argument('--lambda_anchor', type=float, default=0.1, help='anchor proximal term constant')
    parser.add_argument('--tau', type=float, default=0.5, help='moon temperature parameter')
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--momentum_anchor', type=float, default=0.5, help="dynamic momentum update for feature anchor(default: 0.5)")


    # model and data split arguments
    parser.add_argument('--dims_feature', type=int, default=192, help="feature dimension")#192
    parser.add_argument('--trainset_sample_rate', type=int, default=1, help="trainset sample rate")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_perclass', type=int, default=10, help="number of per class in one client dataset")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--split', type=str, default='similarity_2CNN_2', help='dataset spliting setting')
    parser.add_argument('--skew', type=str, default='label', help='distribution skew setting')


    # other arguments
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 1)')

    
    args = parser.parse_args(args=[])

    return args

