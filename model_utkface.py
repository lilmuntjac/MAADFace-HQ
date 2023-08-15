import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import UTKFace
from src.models import CategoricalModel
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset, dataloader (UTKFace)
    utkface = UTKFace(batch_size=args.batch_size)
    train_dataloader = utkface.train_dataloader
    val_dataloader = utkface.val_dataloader

    # model, optimizer, and scheduler
    print(f'Calling model predicting gender and age')
    model = CategoricalModel(out_feature=11, weights=None).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    model_stat_path = Path(args.model_stat_root)/args.model_name
    if args.resume:
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, name=args.resume, root_folder=model_ckpt_path)
        train_stat = load_stats(name=args.model_name+'_train', root_folder=model_stat_path)
        val_stat = load_stats(name=args.model_name+'_val', root_folder=model_stat_path)
    else:
        train_stat, val_stat = np.array([]), np.array([])
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')

    # def to_prediction(logit):
    #     _, race_pred = torch.max(logit[:,0:5],  dim=1)
    #     _, gender_pred = torch.max(logit[:,5:7], dim=1)
    #     _, age_pred = torch.max(logit[:,7:16], dim=1)
    #     pred = torch.stack((race_pred, gender_pred, age_pred), dim=1)
    #     return pred
    def to_prediction(logit):
        _, gender_pred = torch.max(logit[:,0:2], dim=1)
        _, age_pred = torch.max(logit[:,2:11], dim=1)
        pred = torch.stack((gender_pred, age_pred), dim=1)
        return pred

    def train():
        train_stat = np.array([])
        model.train()
        # training loop
        for batch_idx, (data, raw_label) in enumerate(train_dataloader):
            # sens, label = raw_label[:,0:1], raw_label
            sens, label = raw_label[:,0:1], raw_label[:,1:]
            data, label, sens = data.to(device), label.to(device), sens.to(device)
            instance = normalize(data)
            optimizer.zero_grad()
            logit = model(instance)
            # loss = F.cross_entropy(logit[:, 0:5], label[:,0]) + \
            #        F.cross_entropy(logit[:, 5:7], label[:,1]) + \
            #        F.cross_entropy(logit[:, 7:16], label[:,2])
            loss = F.cross_entropy(logit[:, 0:2], label[:,0]) + \
                   F.cross_entropy(logit[:, 2:11], label[:,1])
            loss.backward()
            optimizer.step()
            # collecting performance information
            pred = to_prediction(logit)
            stat = calc_groupacc(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, ?, 4), ?: race + gender + age, 4: split in 2 groups x (right and wrong)

    def val():
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, raw_label) in enumerate(val_dataloader):
                # sens, label = raw_label[:,0:1], raw_label
                sens, label = raw_label[:,0:1], raw_label[:,1:]
                data, label, sens = data.to(device), label.to(device), sens.to(device)
                instance = normalize(data)
                logit = model(instance)
                # collecting performance information
                pred = to_prediction(logit)
                stat = calc_groupacc(pred, label, sens)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, ?, 4), ?: race + gender + age, 4: split in 2 groups x (right and wrong)
    # summarize the status in validation set for some adjustment
    def get_stats_per_epoch(stat):
        # Input: statistics for a single epochs, shape (1, ?, 4)
        group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [stat[0,:,i] for i in range(0, 4)]
        group_1_acc = group_1_correct/(group_1_correct+group_1_wrong)
        group_2_acc = group_2_correct/(group_2_correct+group_2_wrong)
        total_acc = (group_1_correct+group_2_correct)/(group_1_correct+group_1_wrong+group_2_correct+group_2_wrong)
        acc_diff = abs(group_1_acc-group_2_acc)
        stat_dict = {"group_1_acc": group_1_acc, "group_2_acc": group_2_acc, 
                     "total_acc": total_acc, "acc_diff": acc_diff}
        return stat_dict
    # print the epoch status on to the terminal
    def show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch):
        # attr_list = ["Race", "Gender", "Age"]
        attr_list = ["Gender", "Age"]
        for index, attr_name in enumerate(attr_list):
            print(f'    attribute: {attr_name: >40}')
            stat_dict = get_stats_per_epoch(train_stat_per_epoch)
            group_1_acc, group_2_acc = stat_dict["group_1_acc"][index], stat_dict["group_2_acc"][index]
            total_acc, acc_diff = stat_dict["total_acc"][index], stat_dict["acc_diff"][index]
            print(f'    train    {group_1_acc:.4f} - {group_2_acc:.4f} - {total_acc:.4f} -- {acc_diff:.4f}')
            stat_dict = get_stats_per_epoch(val_stat_per_epoch)
            group_1_acc, group_2_acc = stat_dict["group_1_acc"][index], stat_dict["group_2_acc"][index]
            total_acc, acc_diff = stat_dict["total_acc"][index], stat_dict["acc_diff"][index]
            print(f'    val      {group_1_acc:.4f} - {group_2_acc:.4f} - {total_acc:.4f} -- {acc_diff:.4f}')
        print(f'')

    # Run the code
    print(f'Start training model')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        train_stat_per_epoch = train()
        # scheduler.step()
        val_stat_per_epoch = val()
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch:4} done in {epoch_time/60:.4f} mins')
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # print some basic statistic
        show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch)
        # save model checkpoint
        save_model(model, optimizer, scheduler, name=f'{epoch:04d}', root_folder=model_ckpt_path)
    # save basic statistic
    save_stats(train_stat, f'{args.model_name}_train', root_folder=model_stat_path)
    save_stats(val_stat, f'{args.model_name}_val', root_folder=model_stat_path)
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Model training")
    # Training related arguments
    parser.add_argument("--seed", default=32138, type=int, help="seed for the model training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-1, type=float, help="step size for model training")
    # For model trained
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-stat-root", default='/tmp2/npfe/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--resume", default="", help="name of a checkpoint, without .pth")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the weight loaded")

    return parser


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)
