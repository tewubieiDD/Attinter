import os
import time

import torch
from torch import nn

from sklearn.metrics import roc_auc_score as ras


def get_optimizer_param_groups(model, weight_decay, no_decay_classes=None):
    manifold_params = []
    standard_params = []

    no_decay_param_ids = set()

    if no_decay_classes is not None:
        for module in model.modules():
            if isinstance(module, no_decay_classes):
                for param in module.parameters():
                    if param.requires_grad:
                        no_decay_param_ids.add(id(param))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_parametrized = "parametrizations" in name

        if id(param) in no_decay_param_ids or is_parametrized:
            manifold_params.append(param)
        else:
            standard_params.append(param)

    # print(f"Standard params (wd={weight_decay}): {len(standard_params)} tensors")
    # print(f"Manifold params (wd=0.0): {len(manifold_params)} tensors")

    return [
        {'params': standard_params, 'weight_decay': weight_decay},
        {'params': manifold_params, 'weight_decay': 0.0}
    ]


def train_network_loss(net, train_loader, val_loader, test_loader, recorder, device='cpu', **kwargs):
    device = torch.device(device)
    epochs = kwargs.get('epochs', 200)
    lr = kwargs.get('lr', 1e-3)
    wd = kwargs.get('wd', 0)
    test_metric_type = kwargs.get('test_metric_type', 'acc')
    writer = getattr(recorder, 'writer', None)

    loss_fn = nn.CrossEntropyLoss().to(device)
    param_groups = get_optimizer_param_groups(net, wd)
    optimizer = torch.optim.Adam(param_groups, lr=lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)

    best_loss = 1e10
    best_test_acc = 0
    final_path = os.path.join(recorder.exp_dir, 'best_model.pt')
    net = net.to(device)

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

        current_test_acc = None
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), final_path)
            recorder.logger.info(f"--> [Checkpoint] Best model saved to {final_path}")

            if test_loader:
                test_func = test_network_auc if test_metric_type == 'auc' else test_network
                current_test_acc = test_func(net, test_loader, device)
                recorder.logger.info(f"test_{test_metric_type}:{current_test_acc}")
            else:
                current_test_acc = val_acc
                recorder.logger.info(f"test_acc(val_acc):{current_test_acc}")

            best_test_acc = current_test_acc

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_time": val_time,
            "test_acc": current_test_acc,
        }
        recorder.log_iteration(epoch_data)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

    if writer is not None:
        writer.close()

    return best_test_acc


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


def train_network_acc(net, train_loader, val_loader, test_loader, recorder, device='cpu', **kwargs):
    device = torch.device(device)
    epochs = kwargs.get('epochs', 200)
    lr = kwargs.get('lr', 1e-3)
    wd = kwargs.get('wd', 0)
    lr_step_size = kwargs.get('lr_step_size', 50)
    lr_gamma = kwargs.get('lr_gamma', 0.8)
    min_lr = kwargs.get('min_lr', 0.0)
    test_metric_type = kwargs.get('test_metric_type', 'acc')
    writer = getattr(recorder, 'writer', None)

    loss_fn = nn.CrossEntropyLoss().to(device)
    param_groups = get_optimizer_param_groups(net, wd)
    optimizer = torch.optim.Adam(param_groups, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)

    best_acc = 0.0
    best_test_acc = 0.0
    final_path = os.path.join(recorder.exp_dir, 'best_model.pt')
    net = net.to(device)

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

        current_test_acc = None
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), final_path)
            recorder.logger.info(f"--> [Checkpoint] Best model saved to {final_path}")

            if test_loader:
                test_func = test_network_auc if test_metric_type == 'auc' else test_network
                current_test_acc = test_func(net, test_loader, device)
                recorder.logger.info(f"test_{test_metric_type}:{current_test_acc}")
            else:
                current_test_acc = val_acc
                recorder.logger.info(f"test_acc(val_acc):{current_test_acc}")

            best_test_acc = current_test_acc

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_time": val_time,
            "test_acc": current_test_acc,
        }
        recorder.log_iteration(epoch_data)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

        # scheduler.step()
        # for group in optimizer.param_groups:
        #     group['lr'] = max(group['lr'], min_lr)

    if writer is not None:
        writer.close()

    return best_test_acc
