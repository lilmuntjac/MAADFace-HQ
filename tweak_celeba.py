import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import CelebA
from src.models import BinaryModel
from src.tweaker import Tweaker, Losses
from src.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset, dataloader (CelebA)
    all_attr_list = args.attr_list.copy()
    all_attr_list.append("Male") # add the sensitive atttribute
    celeba = CelebA(batch_size=args.batch_size, attr_list=all_attr_list)
    train_dataloader = celeba.train_dataloader
    val_dataloader = celeba.val_dataloader

    # the base model, optimizer, and scheduler
    attr_count = len(args.attr_list)
    print(f'Calling model capable of predicting {attr_count} attributes.')
    model = BinaryModel(out_feature=attr_count, weights=None).to(device)
    _optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=30, gamma=0.1)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    load_model(model, _optimizer, _scheduler, name=args.model_ckpt_name, root_folder=model_ckpt_path)

    # tweaking element
    advatk_ckpt_path = Path(args.advatk_ckpt_root)/args.advatk_name
    advatk_stat_path = Path(args.advatk_stat_root)/args.advatk_name
    match args.adv_type:
        case "noise" | "patch":
            adv_component = torch.full((1, 3, 224, 224), 0.0).to(device)
        case "frame" | "eyeglasses":
            adv_component = torch.full((1, 3, 224, 224), 0.5).to(device)
        case _:
            assert False, "Unknown element type"
    tweaker = Tweaker(batch_size=args.batch_size, tweak_type=args.adv_type)
    losses = Losses(loss_type=args.loss_type, fairness_criteria=args.fairness_matrix, 
                    pred_type='binary', soft_label=False)

    if args.resume:
        adv_component = load_stats(name=args.resume, root_folder=advatk_ckpt_path)
        adv_component = torch.from_numpy(adv_component).to(device)
        train_stat = load_stats(name='train', root_folder=advatk_stat_path)
        val_stat = load_stats(name='val', root_folder=advatk_stat_path)
    else:
        train_stat, val_stat = np.array([]), np.array([])
    adv_component = nn.Parameter(adv_component)
    adversary_optimizer = torch.optim.SGD([adv_component], lr=args.lr, momentum=1e-6)
    adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1, gamma=0.9)
    p_coef = torch.tensor(args.p_coef).to(device)
    n_coef = torch.tensor(args.n_coef).to(device)
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')
    
    def to_prediction(logit):
        # conert binary logit into prediction
        pred = torch.where(logit > 0.5, 1, 0)
        return pred

    # train and validation function
    def train():
        train_stat = np.array([])
        model.eval()
        # training loop
        for batch_idx, (data, raw_label) in enumerate(train_dataloader):
            raw_label = raw_label.to(torch.float32)
            data, raw_label = data.to(device), raw_label.to(device)
            # tweak on data
            data, raw_label = tweaker.apply(data, raw_label, adv_component)
            label, sens = raw_label[:,:-1], raw_label[:,-1:None]
            instance = normalize(data)
            adversary_optimizer.zero_grad()
            logit = model(instance)
            loss = losses.run(logit, label, sens)
            loss.backward()
            # if batch_idx % 128 ==0:
            #     adversary_optimizer.step()
            #     tweaker.retify(adv_component)
            #     adversary_optimizer.zero_grad()
            adversary_optimizer.step()
            tweaker.retify(adv_component)
            # collecting performance information
            pred = to_prediction(logit)
            stat = calc_groupcm_soft(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, attribute, 8)
    
    def val(dataloader=val_dataloader):
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, raw_label) in enumerate(dataloader):
                data, raw_label = data.to(device), raw_label.to(device)
                # tweak on data
                data, raw_label = tweaker.apply(data, raw_label, adv_component)
                label, sens = raw_label[:,:-1], raw_label[:,-1:None]
                instance = normalize(data)
                logit = model(instance)
                # collecting performance information
                pred = to_prediction(logit)
                stat = calc_groupcm_soft(pred, label, sens)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, attribute, 8)
    # summarize the status in validation set for some adjustment
    def get_stats_per_epoch(stat):
        # Input: statistics for a single epochs, shape (1, attributes, 8)
        mtp, mfp, mfn, mtn = [stat[0,:,i] for i in range(0, 4)]
        ftp, ffp, ffn, ftn = [stat[0,:,i] for i in range(4, 8)]
        # Accuracy
        macc = (mtp+mtn)/(mtp+mfp+mfn+mtn)
        facc = (ftp+ftn)/(ftp+ffp+ffn+ftn)
        tacc = (mtp+mtn+ftp+ftn)/(mtp+mfp+mfn+mtn+ftp+ffp+ffn+ftn)
        # Fairness
        mtpr, mtnr = mtp/(mtp+mfn), mtn/(mtn+mfp)
        ftpr, ftnr = ftp/(ftp+ffn), ftn/(ftn+ffp)
        tpr_diff, tnr_diff = abs(mtpr-ftpr), abs(mtnr-ftnr)
        equality_of_opportunity = tpr_diff
        equalized_odds = tpr_diff+tnr_diff
        stat_dict = {"male_acc": macc, "female_acc": facc, "total_acc": tacc,
                     "tpr_diff": tpr_diff, "tnr_diff": tnr_diff, 
                     "equality_of_opportunity": equality_of_opportunity, "equalized_odds": equalized_odds}
        return stat_dict
    # print the epoch status on to the terminal
    def show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch):
        for index, attr_name in enumerate(args.attr_list):
            print(f'    attribute: {attr_name: >40}')
            stat_dict = get_stats_per_epoch(train_stat_per_epoch)
            macc, facc, tacc = stat_dict["male_acc"][index], stat_dict["female_acc"][index], stat_dict["total_acc"][index]
            equality_of_opportunity, equalized_odds = stat_dict["equality_of_opportunity"][index], stat_dict["equalized_odds"][index]
            print(f'    train    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
            stat_dict = get_stats_per_epoch(val_stat_per_epoch)
            macc, facc, tacc = stat_dict["male_acc"][index], stat_dict["female_acc"][index], stat_dict["total_acc"][index]
            equality_of_opportunity, equalized_odds = stat_dict["equality_of_opportunity"][index], stat_dict["equalized_odds"][index]
            print(f'    val      {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
        print(f'')

    # Run the code
    print(f'Start training model')
    start_time = time.time()

    if not args.resume:
        empty_time = time.time()
        print(f'collecting statistic for empty tweaks')
        train_stat_per_epoch = val(train_dataloader)
        val_stat_per_epoch = val()
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        show_stats_per_epoch(train_stat_per_epoch, val_stat_per_epoch)
        print(f'done in {(time.time()-empty_time)/60:.4f} mins')
    # some parameter might needs the init stats

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
        # save the adversarial component for each epoch
        component = adv_component.detach().cpu().numpy()
        save_stats(component, f'{epoch:04d}', root_folder=advatk_ckpt_path)
    # save basic statistic
    save_stats(train_stat, f'train', root_folder=advatk_stat_path)
    save_stats(val_stat, f'val', root_folder=advatk_stat_path)
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')
            
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Tweak the model by data pre-processing")
    # For base model loaded
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    # parser.add_argument("--model-stat-root", default='/tmp2/npfe/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--model-ckpt-name", default='default_model', type=str, help='name for the model checkpoint, without .pth')
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")

    # For input tweaking element
    parser.add_argument("--advatk-ckpt-root", default='/tmp2/npfe/advatk', type=str, help='root path for adversarial atttack statistic')
    parser.add_argument("--advatk-stat-root", default='/tmp2/npfe/advatk_stats', type=str, help='root path for adversarial attack itself')
    parser.add_argument("--advatk-name", default='default_advatk', type=str, help='name for the advatk trained')
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check on the element loaded")
    # training related
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-1, type=float, help="step size for model training")
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    # setting for each types of tweek




    # binary model
    parser.add_argument("--fairness-matrix", default="prediction quaility", help="how to measure fairness")
    parser.add_argument("--p-coef", default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1,], type=float, nargs='+', help="coefficient multiply on positive recovery loss, need to be match with the number of attributes")
    parser.add_argument("--n-coef", default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1,], type=float, nargs='+', help="coefficient multiply on negative recovery loss, need to be match with the number of attributes")

    # loss types
    parser.add_argument("--loss-type", default='direct', type=str, help="Type of loss used")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)