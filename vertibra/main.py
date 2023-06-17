import argparse
import train
import test_e
import time

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_last.pth', help='weights to be resumed')
    parser.add_argument('--data_dir', type=str, default='Datasets\spinal', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        is_object = train.Network(args)
        is_object.train_network(args)

    elif args.phase == 'test':
        is_object = test_e.Network(args)
        is_object.eval(args, save=False)    
        is_object = test_e.Network1(args)
        is_object.test(args, save=False)    