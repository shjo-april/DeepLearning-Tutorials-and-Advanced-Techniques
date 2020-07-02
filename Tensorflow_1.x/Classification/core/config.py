
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='flower_photos', type=str)
    parser.add_argument('--use_gpu', default='0', type=str)
    
    parser.add_argument('--optimizer', default='momentum', type=str)
    
    parser.add_argument('--learning_rate', default=0.016, type=float)
    parser.add_argument('--learning_rate_schedule', default='cosine_annealing', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--image_size', default=224, type=int)

    parser.add_argument('--augmentation', default='Default', type=str)

    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--validation_epochs', default=10, type=int)

    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_config()
    print(args.max_epoch)

