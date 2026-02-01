import argparse
import os
import copy
import socket
from contextlib import closing

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


def _find_free_port() -> int:
    # Safest local method: ask OS for an ephemeral port
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def _setup_ddp(rank: int, world_size: int, master_addr: str, master_port: int):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank: int, world_size: int, master_addr: str, master_port: int, args):
    is_ddp = world_size > 1

    if is_ddp:
        _setup_ddp(rank, world_size, master_addr, master_port)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeds: deterministic enough, but different ranks won't fight over identical shuffles
    torch.manual_seed(args.seed + rank)

    model = SRCNN().to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params": model.module.conv1.parameters() if is_ddp else model.conv1.parameters()},
            {"params": model.module.conv2.parameters() if is_ddp else model.conv2.parameters()},
            {
                "params": model.module.conv3.parameters() if is_ddp else model.conv3.parameters(),
                "lr": args.lr * 0.1,
            },
        ],
        lr=args.lr,
    )

    train_dataset = TrainDataset(args.train_file)

    # Each GPU gets its own shard; shuffle must be controlled by sampler each epoch
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None

    # Avoid multiplying workers by GPUs -> keep total roughly stable
    num_workers = args.num_workers
    if is_ddp and args.num_workers > 0:
        num_workers = max(1, args.num_workers // world_size)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    best_weights = copy.deepcopy((model.module if is_ddp else model).state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = AverageMeter()

        # Only rank0 shows progress bar to avoid terminal spam
        total = len(train_dataset) - (len(train_dataset) % args.batch_size)
        pbar = tqdm(total=total, disable=(is_ddp and rank != 0))
        pbar.set_description(f"epoch: {epoch}/{args.num_epochs - 1}")

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(inputs)
            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), inputs.size(0))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if not (is_ddp and rank != 0):
                pbar.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                pbar.update(inputs.size(0))

        pbar.close()

        # Save checkpoints only on rank0
        if (not is_ddp) or rank == 0:
            state = (model.module if is_ddp else model).state_dict()
            torch.save(state, os.path.join(args.outputs_dir, f"epoch_{epoch}.pth"))

        # Eval only on rank0 (simple + safe)
        if (not is_ddp) or rank == 0:
            model.eval()
            epoch_psnr = AverageMeter()

            with torch.no_grad():
                for inputs, labels in eval_dataloader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(inputs).clamp(0.0, 1.0)
                    epoch_psnr.update(calc_psnr(preds, labels), inputs.size(0))

            print(f"eval psnr: {epoch_psnr.avg:.2f}")

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy((model.module if is_ddp else model).state_dict())

    if (not is_ddp) or rank == 0:
        print(f"best epoch: {best_epoch}, psnr: {best_psnr:.2f}")
        torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))

    if is_ddp:
        _cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, f"x{args.scale}")
    os.makedirs(args.outputs_dir, exist_ok=True)

    cudnn.benchmark = True

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size > 1:
        master_addr = "127.0.0.1"
        master_port = _find_free_port()
        mp.spawn(
            main_worker,
            args=(world_size, master_addr, master_port, args),
            nprocs=world_size,
            join=True,
        )
    else:
        main_worker(rank=0, world_size=1, master_addr="127.0.0.1", master_port=0, args=args)
