from __future__ import annotations

import os
import time
from typing import Callable, Optional

import torch
from torch import nn


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

    return [
        {"params": standard_params, "weight_decay": weight_decay},
        {"params": manifold_params, "weight_decay": 0.0},
    ]


from sklearn.metrics import balanced_accuracy_score


def _x_y_d(batch):
    if isinstance(batch, dict):
        return batch["x"], batch.get("y"), batch.get("d")
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2 and isinstance(batch[0], dict):
            return batch[0]["x"], batch[1], batch[0].get("d")
        return batch[0], batch[1] if len(batch) > 1 else None, batch[2] if len(batch) > 2 else None
    return batch, None, None


def _logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _bacc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred) if y_true else 0.0


def _evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_loss, total_n = 0.0, 0
    truth, pred = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, d = _x_y_d(batch)
            if y is None:
                raise ValueError("The source validation loader must provide labels.")
            x, y = x.to(device), y.to(device)
            out = _logits(model(x, d)) if d is not None else _logits(model(x))
            loss = loss_fn(out, y)
            n = int(y.numel())
            total_loss += loss.item() * n
            total_n += n
            truth.extend(y.cpu().tolist())
            pred.extend(out.argmax(1).cpu().tolist())
    return total_loss / total_n if total_n else 0.0, _bacc(truth, pred)


def _collect_target_x(target_loader, device):
    """Collect target inputs only; labels are deliberately not accessed."""
    xs = []
    for batch in target_loader:
        x, _, _ = _x_y_d(batch)
        xs.append(x.to(device))
    if not xs:
        raise ValueError("The target loader is empty.")
    return torch.cat(xs, dim=0)


def _matrix_power(x, exponent, eps=1e-6):
    eigval, eigvec = torch.linalg.eigh(x)
    eigval = eigval.clamp_min(eps).pow(exponent)
    return (eigvec * eigval.unsqueeze(-2)) @ eigvec.transpose(-1, -2)


def _airm_distance_sq(a, b, eps=1e-6):
    a_inv_sqrt = _matrix_power(a, -0.5, eps)
    eigval = torch.linalg.eigvalsh(a_inv_sqrt @ b @ a_inv_sqrt).clamp_min(eps)
    return torch.log(eigval).square().sum(-1)


def _frechet_mean(x, steps=8, eps=1e-6):
    """Karcher-flow approximation of the AIRM Fr茅chet mean."""
    mean = x[0]
    for _ in range(max(1, steps)):
        inv_sqrt = _matrix_power(mean, -0.5, eps)
        sqrt = _matrix_power(mean, 0.5, eps)
        tangent = inv_sqrt @ x @ inv_sqrt
        eigval, eigvec = torch.linalg.eigh(tangent)
        log_tangent = (eigvec * eigval.clamp_min(eps).log().unsqueeze(-2)) @ eigvec.transpose(-1, -2)
        mean = sqrt @ _matrix_exp(log_tangent.mean(0)) @ sqrt
    return mean


def _matrix_exp(x):
    eigval, eigvec = torch.linalg.eigh(x)
    return (eigvec * eigval.exp().unsqueeze(-2)) @ eigvec.transpose(-1, -2)


def fit_target_stats(model, target_x, *, frechet_steps=8, eps=1e-6):
    """Fit target statistics from unlabeled target data.

    The model must expose ``extract_spd(x)`` and ``set_target_stats(mean,
    variance)``.  ``extract_spd`` should return latent SPD features with shape
    ``(N, D, D)``.  The callback computes the target Fr茅chet mean ``G_t`` and
    variance ``nu_t^2`` and stores them through ``set_target_stats``.
    """
    if not callable(getattr(model, "extract_spd", None)):
        raise AttributeError("Model must implement extract_spd(x) for SPDDSMBN adaptation.")
    if not callable(getattr(model, "set_target_stats", None)):
        raise AttributeError("Model must implement set_target_stats(mean, variance).")
    model.eval()
    with torch.no_grad():
        z = model.extract_spd(target_x)
        if z.ndim != 3 or z.shape[-1] != z.shape[-2]:
            raise ValueError("extract_spd(x) must return a tensor shaped (N, D, D).")
        mean = _frechet_mean(z, steps=frechet_steps, eps=eps)
        variance = _airm_distance_sq(mean.unsqueeze(0), z, eps=eps).mean()
        model.set_target_stats(mean, variance)
    return {"mean": mean, "variance": variance, "n_target": int(z.shape[0])}


def adapt_target_domain(model, target_loader, device="cpu"):
    """Official-style target adaptation: REFIT each domain, then BUFFER."""
    device = torch.device(device)
    model.eval()
    batches = []
    for batch in target_loader:
        x, _y, d = _x_y_d(batch)
        if d is None:
            raise ValueError("Target batches must contain domain ids d.")
        batches.append((x.to(device), d.to(device)))
    if not batches:
        raise ValueError("The target loader is empty.")
    if not callable(getattr(model, "domainadapt_finetune", None)):
        raise AttributeError("Model must implement official domainadapt_finetune(x, y, d, target_domains).")
    # Official code calls the model once per complete domain, not once per
    # minibatch.  Reassemble all target samples by domain before REFIT.
    domain_ids = torch.cat([d for _, d in batches]).unique()
    with torch.no_grad():
        for domain in domain_ids:
            x_domain = torch.cat([x[d == domain] for x, d in batches], dim=0)
            d_domain = torch.full((x_domain.shape[0],), int(domain), dtype=torch.long, device=device)
            model.domainadapt_finetune(
                x=x_domain,
                y=None,
                d=d_domain,
                target_domains=domain_ids,
            )
    return {"mode": "REFIT->BUFFER", "n_target": sum(x.shape[0] for x, _ in batches)}

def predict_target(model, target_loader, device="cpu", predict_fn: Optional[Callable] = None):
    """Predict with frozen target statistics; returns predictions and labels."""
    model.eval()
    device = torch.device(device)
    predict = predict_fn or predict_fn_default
    predictions, labels = [], []
    with torch.no_grad():
        for batch in target_loader:
            x, y, d = _x_y_d(batch)
            out = _logits(predict(model, x.to(device), d.to(device) if torch.is_tensor(d) else d))
            predictions.extend(out.argmax(1).cpu().tolist())
            if y is not None:
                labels.extend(y.cpu().tolist())
    return predictions, labels


def predict_fn_default(model, x, d=None):
    """Paper ``BUFFER`` prediction using fixed target statistics."""
    if d is None:
        return model(x)
    return model(x, d)


# Public name retained for callers that want to override the BUFFER forward.
predict_fn = predict_fn_default


def train_network_loss(net, train_loader, val_loader, test_loader, recorder, device="cpu", **kwargs):
    device = torch.device(device)
    epochs = int(kwargs.get("epochs", 200))
    lr = float(kwargs.get("lr", 1e-3))
    wd = float(kwargs.get("wd", 0.0))
    frechet_steps = int(kwargs.get("frechet_steps", 8))
    eps = float(kwargs.get("eps", 1e-6))
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer_cls = kwargs.get("optimizer_cls", torch.optim.Adam)
    optimizer = optimizer_cls(get_optimizer_param_groups(net, wd), lr=lr)
    best_val_loss = float("inf")
    best_path = os.path.join(recorder.exp_dir, "best_model.pt") if recorder is not None else None
    logger = getattr(recorder, "logger", None)
    writer = getattr(recorder, "writer", None)

    for epoch in range(epochs):
        net.train()
        total_loss, n = 0.0, 0
        truth, pred = [], []
        start = time.time()
        for batch in train_loader:
            x, y, d = _x_y_d(batch)
            if y is None:
                raise ValueError("The source training loader must provide labels.")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = _logits(net(x, d)) if d is not None else _logits(net(x))
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            bs = int(y.numel())
            total_loss += loss.item() * bs
            n += bs
            truth.extend(y.detach().cpu().tolist())
            pred.extend(out.detach().argmax(1).cpu().tolist())

        train_loss = total_loss / n if n else 0.0
        train_bacc = _bacc(truth, pred)
        val_loss, val_bacc = _evaluate(net, val_loader, device)
        if logger:
            logger.info(
                f"Iteration{epoch + 1}===== train_loss:{train_loss:.4f} "
                f"val_loss:{val_loss:.4f} train_bacc:{train_bacc:.4f} "
                f"val_bacc:{val_bacc:.4f} train_time:{time.time() - start:.4f}"
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_path:
                torch.save(net.state_dict(), best_path)
        if recorder is not None and hasattr(recorder, "log_iteration"):
            recorder.log_iteration({"epoch": epoch + 1, "train_loss": train_loss,
                                    "train_bacc": train_bacc, "val_loss": val_loss,
                                    "val_bacc": val_bacc})
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("BalancedAccuracy/train", train_bacc, epoch + 1)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
            writer.add_scalar("BalancedAccuracy/val", val_bacc, epoch + 1)

    if best_path and os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=device))
    adaptation = adapt_target_domain(net, test_loader, device)
    predictor = kwargs.get("predict_fn", predict_fn_default)
    predictions, labels = predict_target(net, test_loader, device, predictor)
    target_bacc = _bacc(labels, predictions) if labels else None
    if logger:
        logger.info(f"target_adaptation:{adaptation}")
        if target_bacc is not None:
            logger.info(f"target_bacc:{target_bacc:.4f}")
    if writer is not None:
        writer.close()
    return target_bacc, {"adaptation": adaptation, "predictions": predictions, "labels": labels}

