
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='UCF-101', type=str)
    parser.add_argument('--use_gpu', default='0', type=str)
    
    parser.add_argument('--optimizer', default='adam', type=str)
    
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_schedule', default='step_decays', type=str)
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    
    parser.add_argument('--augmentation', default='Default', type=str)

    parser.add_argument('--the_number_of_frame', default=5, type=int)

    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--validation_epochs', default=10, type=int)

    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_config()
    print(args.max_epoch)

