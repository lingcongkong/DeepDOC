import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--model_name', type=str, default='DeepDOC')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    

    # MODEL PARAMETER
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model_depth', type=float, default=50)
    

    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)


    config = parser.parse_args()
    return config