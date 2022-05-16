import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross_validation_with_val_set(
    dataset,
    model,
    folds,
    epochs,
    batch_size,
    lr,
    lr_decay_factor,
    lr_decay_step_size,
    weight_decay,
    use_tqdm=True,
    writer=None,
    logger=None,
):

    val_losses, accs, durations = [], [], []

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset.data.subgraph(dataset.data.train_mask)
        test_dataset = dataset.data.subgraph(dataset.data.test_mask)
        val_dataset = dataset.data.subgraph(dataset.data.val_mask)

        if "adj" in train_dataset:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        if use_tqdm:
            t = tqdm(total=epochs, desc="Fold #" + str(fold))
        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_loss = eval_loss(model, val_loader)
            val_losses.append(val_loss)
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_losses[-1],
                "test_acc": accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if writer is not None:
                writer.add_scalar(f"Fold{fold}/Train_Loss", train_loss, epoch)
                writer.add_scalar(f"Fold{fold}/Val_Loss", val_loss, epoch)
                writer.add_scalar(
                    f"Fold{fold}/Lr", optimizer.param_groups[0]["lr"], epoch
                )

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_decay_factor * param_group["lr"]

            if use_tqdm:
                t.set_postfix(
                    {
                        "Train_Loss": "{:05.3f}".format(train_loss),
                        "Val_Loss": "{:05.3f}".format(val_loss),
                    }
                )
                t.update(1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(
        "Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}".format(
            loss_mean, acc_mean, acc_std, duration_mean
        )
    )
    if writer is not None:
        writer.add_scalar(f"Final/Test_Acc", acc_mean, epoch)
        writer.add_scalar(f"Final/Test_Acc_Std", acc_std, epoch)
        writer.add_scalar(f"Final/Test_Loss", loss_mean, epoch)
        writer.close()

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    # skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    # test_indices, train_indices = [], []
    # for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
    #     test_indices.append(torch.from_numpy(idx))
    #
    # val_indices = [test_indices[i - 1] for i in range(folds)]
    #
    # for i in range(folds):
    #     train_mask = torch.ones(len(dataset), dtype=torch.bool)
    #     train_mask[test_indices[i]] = 0
    #     train_mask[val_indices[i]] = 0
    #     train_indices.append(train_mask.nonzero().view(-1))
    return [0]*folds, [0]*folds, [0]*folds
    # return dataset.data.train_mask, dataset.data.test_mask, dataset.data.val_mask


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    optimizer.zero_grad()
    data = loader.dataset
    out = model(data)
    y = data.y.view(-1).to(data.x.device)
    loss = F.cross_entropy(out, y.cuda())
    loss.backward()
    total_loss += loss.item() * num_graphs(data)
    optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    data = loader.dataset
    with torch.no_grad():
        pred = model(data).max(1)[1].to(data.x.device)
    correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    data = loader.dataset
    with torch.no_grad():
        out = model(data)
    loss += F.cross_entropy(out.to(data.x.device), data.y.view(-1), reduction="sum").item()
    return loss / len(loader.dataset)
