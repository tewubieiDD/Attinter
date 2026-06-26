import os
import time

import torch
from torch import nn

from sklearn.metrics import roc_auc_score as ras


def train_network(net, train_loader, val_loader, test_loader, recorder=None, device='cpu', **kwargs):
    device = torch.device(device)
    epochs = kwargs.get('epochs', 200)
    lr = kwargs.get('lr', 1e-3)
    wd = kwargs.get('wd', 0)
    save_strategy = kwargs.get('save_strategy', 'best')
    eval_criterion = kwargs.get('eval_criterion', 'loss')
    test_metric_type = kwargs.get('test_metric_type', 'acc')

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)

    bestLoss = 1e10
    bestAcc = 0.0
    test_acc = 0
    final_path = os.path.join(recorder.exp_dir, 'best_model.pt')
    net = net.to(device)

    if eval_criterion == 'loss':
        writer = None
    else:
        writer = recorder.writer

    for epoch in range(epochs):
        net.train()
        loss_tr, acc_tr, tr_len = 0.0, 0, 0
        tr_start = time.time()
        for xb, yb in train_loader:
            bs = xb.shape[0]
            tr_len += bs
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            acc_tr += (out.argmax(1) == yb).sum().item()
            loss_tr += loss.item() * bs

        train_time = time.time() - tr_start
        train_loss = loss_tr / tr_len if tr_len > 0 else 0.0
        train_acc = acc_tr / tr_len if tr_len > 0 else 0.0

        net.eval()
        loss_val, acc_val, val_len = 0.0, 0, 0
        val_start = time.time()
        with torch.no_grad():
            for xb, yb in val_loader:
                bs = xb.shape[0]
                val_len += bs
                xb, yb = xb.to(device), yb.to(device)

                out = net(xb)
                loss = loss_fn(out, yb)

                acc_val += (out.argmax(1) == yb).sum().item()
                loss_val += loss.item() * bs

        val_time = time.time() - val_start
        val_loss = loss_val / val_len if val_len > 0 else 0.0
        val_acc = acc_val / val_len if val_len > 0 else 0.0

        recorder.logger.info('')
        recorder.logger.info(f'Iteration{epoch + 1}=====')
        recorder.logger.info(f'train_loss:{train_loss:.4f}    val_loss:{val_loss:.4f}    train_time:{train_time:.4f}')
        recorder.logger.info(f'train_acc:{train_acc:.4f}    val_acc:{val_acc:.4f}    val_time:{val_time:.4f}')

        is_best = False
        if eval_criterion == 'loss' and val_loss < bestLoss:
            bestLoss, is_best = val_loss, True
        elif eval_criterion == 'acc' and val_acc > bestAcc:
            bestAcc, is_best = val_acc, True

        trigger_action = (save_strategy == 'all') or (save_strategy == 'best' and is_best)
        if trigger_action:
            if save_strategy == 'all':
                save_path = os.path.join(recorder.exp_dir, 'model', f'net-epoch-{epoch + 1}.pt')
                torch.save(net.state_dict(), save_path)
                if is_best:
                    torch.save(net.state_dict(), final_path)
            else:
                torch.save(net.state_dict(), final_path)
            recorder.logger.info(f"--> [Checkpoint] Best model saved to {final_path}")

            if test_loader is None:
                test_acc = val_acc
                log_msg = f"test_acc: {test_acc}"
            else:
                test_func = test_network_auc if test_metric_type == 'auc' else test_network
                test_acc = test_func(net, test_loader, device)
                log_msg = f"test_{test_metric_type}: {test_acc}"
            recorder.logger.info(f"{log_msg}")

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_time": val_time,
            "test_acc": test_acc,
            "eval_criterion": eval_criterion
        }
        recorder.log_iteration(epoch_data)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

    if recorder and getattr(recorder, 'writer', None):
        recorder.writer.close()

    return test_acc


def test_network(net, test_loader, device):
    net.eval()
    acc_test, test_len = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            bs = xb.shape[0]
            test_len += bs
            xb, yb = xb.to(device), yb.to(device)

            pred = net(xb)
            acc_test += (pred.argmax(1) == yb).sum().item()

    return acc_test / test_len if test_len > 0 else 0.0


def test_network_auc(net, test_loader, device):
    net.eval()
    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    return ras(y_true.detach().numpy(), y_pred.detach().numpy())
