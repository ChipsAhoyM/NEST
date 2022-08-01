from utils import Options
from test import test
from train import train

if __name__ == '__main__':
    args = Options().parse()
    if args.train:
        train()
    else:
        test()