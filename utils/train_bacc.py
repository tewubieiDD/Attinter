import os
import time

import torch
from torch import nn

from sklearn.metrics import balanced_accuracy_score


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


def _balanced_accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return balanced_accuracy_score(y_true, y_pred)


def _append_predictions(y_true, y_pred, logits, labels):
    y_true.extend(labels.detach().cpu().numpy().tolist())
    y_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())


def train_network_loss(net, train_loader, val_loader, test_loader, recorder, device='cpu', **kwargs):
    device = torch.device(device)
    epochs = kwargs.get('epochs', 200)
    lr = kwargs.get('lr', 1e-3)
    wd = kwargs.get('wd', 0)
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
        loss_tr, tr_len = 0.0, 0
        train_true, train_pred = [], []
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

            _append_predictions(train_true, train_pred, out, yb)
            loss_tr += loss.item() * bs
        train_time = time.time() - tr_start
        train_loss = loss_tr / tr_len if tr_len > 0 else 0.0
        train_bacc = _balanced_accuracy(train_true, train_pred)

        net.eval()
        loss_val, val_len = 0.0, 0
        val_true, val_pred = [], []
        val_start = time.time()
        with torch.no_grad():
            for xb, yb in val_loader:
                bs = xb.shape[0]
                val_len += bs
                xb, yb = xb.to(device), yb.to(device)

                out = net(xb)
                loss = loss_fn(out, yb)

                _append_predictions(val_true, val_pred, out, yb)
                loss_val += loss.item() * bs
        val_time = time.time() - val_start
        val_loss = loss_val / val_len if val_len > 0 else 0.0
        val_bacc = _balanced_accuracy(val_true, val_pred)

        recorder.logger.info('')
        recorder.logger.info(f'Iteration{epoch + 1}=====')
        recorder.logger.info(f'train_loss:{train_loss:.4f}    val_loss:{val_loss:.4f}    train_time:{train_time:.4f}')
        recorder.logger.info(f'train_bacc:{train_bacc:.4f}    val_bacc:{val_bacc:.4f}    val_time:{val_time:.4f}')

        current_test_bacc = None
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), final_path)
            recorder.logger.info(f"--> [Checkpoint] Best model saved to {final_path}")

            if test_loader:
                current_test_bacc = test_network(net, test_loader, device)
                recorder.logger.info(f"test_bacc:{current_test_bacc}")
            else:
                current_test_bacc = val_bacc
                recorder.logger.info(f"test_bacc(val_bacc):{current_test_bacc}")

            best_test_acc = current_test_bacc

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_bacc": train_bacc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_bacc": val_bacc,
            "val_time": val_time,
            "test_bacc": current_test_bacc,
        }
        recorder.log_iteration(epoch_data)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('BalancedAccuracy/train', train_bacc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('BalancedAccuracy/val', val_bacc, epoch + 1)

    if writer is not None:
        writer.close()

    return best_test_acc


def test_network(net, test_loader, device):
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = net(xb)
            _append_predictions(y_true, y_pred, pred, yb)

    return _balanced_accuracy(y_true, y_pred)
