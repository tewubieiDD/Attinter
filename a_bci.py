import argparse

from spd.models import build_model
from utils import set_seed, create_exp_dir, Recorder, train_network_loss
from utils.GetBci2a import getAllDataloader

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dataset', type=str, default='MI')
    ap.add_argument('--model', type=str, default='EEGNet5')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    ap.add_argument('--sub', type=int, default=1, help='subjectxx you want to triain')
    ap.add_argument('--epochs', type=int, default=350)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--output_dir', type=str, default='outputs/')
    ap.add_argument('--data_path', type=str, default='data/BCICIV_2a_mat/')
    ap.add_argument('--slice', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--model_path', type=str, default='./checkpoint/bci2a/')
    ap.add_argument('--description', type=str, default='')
    args = vars(ap.parse_args())

    set_seed(args["seed"])
    exp_path = create_exp_dir(args["output_dir"], args["seed"], args["dataset"], args["model"], args["lr"], args["wd"],
                              args["sub"])
    recorder = Recorder(exp_path, args)

    recorder.logger.info(f'subject{args["sub"]}')
    trainloader, validloader, testloader = getAllDataloader(subject=args['sub'],
                                                            ratio=8,
                                                            data_path=args['data_path'],
                                                            bs=args['bs'])

    net = build_model(args['model'], args['dataset'], args).cpu()

    acc = train_network_loss(net, trainloader, validloader, testloader, recorder, **args)
    recorder.save_summary(acc)
    recorder.logger.info(f'{acc * 100:.2f}')
