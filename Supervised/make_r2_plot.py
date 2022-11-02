import argparse
from evaluate_model import evaluate_model

parser = argparse.ArgumentParser(description='Making R2 plots')
parser.add_argument('--fold', default='A', type=str,
                    help='fold to train on')
parser.add_argument('--fraction', default='1', type=str,
                    help='frac to evaluate on')
parser.add_argument('--checkpoint-dir', type=str,
                    help='checkpoint-dir')
parser.add_argument('--plot-title', default='', type=str,
                    help='title of plot')
parser.add_argument('--file-name', default='', type=str,
                    help='name of file saved as png')
parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device to use')

if __name__ == '__main__':
    args = parser.parse_args()
    
    r2 = evaluate_model(checkpoints_dir=args.checkpoint_dir, fold=args.fold, fraction=args.fraction, plot_title=args.plot_title, file_name=args.file_name)
    print('r2: ', r2, '__________________________')
    