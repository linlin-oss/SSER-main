import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model-Master Full Reproducible Framework")

    # data
    parser.add_argument('--dataset', type=str, default='cora', help='Planetoid dataset name')
    parser.add_argument('--train-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--imbalance-ratio', type=float, default=0.9)

    # model
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--mlp-hidden', type=int, default=32)
    parser.add_argument('--att-hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--K', type=int, default=2)

    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if __import__('torch').cuda.is_available() else 'cpu')

    # apg / high-sim
    parser.add_argument('--walk-length', type=int, default=10)
    parser.add_argument('--num-walks', type=int, default=50)
    parser.add_argument('--sim-threshold', type=float, default=0.99995)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)

    # misc
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()
    return args
