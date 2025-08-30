import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from lib.datasets.sub1 import GAMMA_sub1_dataset, GAMMA_sub1_dataset_512, GAMMA_sub1_dataset_512_cup, \
    GAMMA_sub1_dataset_224
import torchvision.transforms as trans
from torch.utils.data import DataLoader, Dataset
from lib.model.dual_resnet import Dual_network34, Dual_network18, Dual_networkx50
import torchvision.models as models
import pdb
from torch.autograd import Variable
import torch
import time
from lib.model.utils.config import get_argparser
import random
import skimage
from lib.losses.Focal_loss import FocalLoss
from lib.model.dual_efficientnet import Dual_efficientnetb0, Dual_efficientnetb1, Dual_efficientnetb2, \
    Dual_efficientnetb3, Dual_efficientnetb4, Dual_efficientnetb5, Dual_efficientnetb6, Dual_efficientnetb7

from lib.model.dual_retfound import Dual_RETFound


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)


def train(args, model, iters, train_loader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    print("Entering training loop...")
    time_start = time.time()
    iter = 0

    if args.model_mode == "18" or args.model_mode == "34" or args.model_mode == 'x50':
        log_file = './logs/resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + time.strftime(
            '%m-%d-%H-%M',
            time.localtime(
                time.time())) + '.txt'
        model_path = './models/dual_resnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode
    else:
        log_file = './logs/efficientnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode + time.strftime(
            '%m-%d-%H-%M',
            time.localtime(
                time.time())) + '.txt'
        model_path = './models/dual_efficientnet' + args.model_mode + '_' + args.sub1dataset_mode + '_' + args.transforms_mode

    model.train()
    model = model.cuda()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    avg_loss_list = []
    avg_kappa_list = []
    best_kappa_100 = 0.

    while iter < iters:
        for fundus_imgs, oct_imgs, labels in train_loader:
            print(f"Training iteration {iter + 1}...")           
            iter += 1
            if iter > iters:
                print("Exiting training loop.")
                break

            optimizer.zero_grad()

            fundus_imgs = Variable(fundus_imgs.cuda())
            oct_imgs = Variable(oct_imgs.cuda())
            labels = Variable(labels.cuda())

            print(f"OCT image shape: {oct_imgs.shape}k")

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)

            print(f"Iteration {iter}: Loss = {loss.item()}")

            for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()

            avg_loss_list.append(loss.item())

            if iter % log_interval == 0:
                avg_loss = np.mean(avg_loss_list)
                avg_kappa_arr = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_arr[:, 0], avg_kappa_arr[:, 1], weights='quadratic')

                avg_loss_list.clear()
                avg_kappa_list.clear()

                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"[TRAIN] iter={iter}/{iters} avg_loss={avg_loss:.4f} avg_kappa={avg_kappa:.4f}\n")

                print(
                    f"[TRAIN] iter={iter}/{iters} avg_loss={avg_loss:.4f} avg_kappa={avg_kappa:.4f}")
                print(f'time cost: {time.time() - time_start:.2f}s')
                time_start = time.time()

            if iter % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"[EVAL] iter={iter}/{iters} avg_loss={avg_loss:.4f} kappa={avg_kappa:.4f}\n")

                print(
                    f"[EVAL] iter={iter}/{iters} avg_loss={avg_loss:.4f} kappa={avg_kappa:.4f}")

                if avg_kappa >= best_kappa_100:
                    best_kappa_100 = avg_kappa
                    print('best_kappa_100:', best_kappa_100)
                    torch.save(model, os.path.join(model_path, f"best_model100_{best_kappa_100:.4f}.pth"))

            model.train()


def val(model, var_loader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        for fundus_imgs, oct_imgs, labels in var_loader:
            fundus_imgs = fundus_imgs.cuda()
            oct_imgs = oct_imgs.cuda()
            labels = labels.cuda()

            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)
            avg_loss_list.append(loss.item())

            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            cache.extend(zip(preds, labels_np))

    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.mean(avg_loss_list)
    return avg_loss, kappa


def get_dataloader(args):
    filelists = os.listdir(args.train_root)
    train_filelists, val_filelists = train_test_split(filelists, test_size=args.val_ratio, random_state=42)

    print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

    if args.model_mode == '34':  # Using RETFound
        img_train_transforms = trans.Compose([
            trans.Resize((224, 224)),  # Resize fundus images to 224x224
            trans.ToTensor()
        ])
        oct_train_transforms = trans.Compose([
            trans.Lambda(lambda x: torch.from_numpy(x).float().div(255))
        ])
        img_val_transforms = trans.Compose([
            trans.Resize((224, 224)),  # Resize fundus images to 224x224
            trans.ToTensor()
        ])
        oct_val_transforms = trans.Compose([
        trans.Lambda(lambda x: torch.from_numpy(np.array(x)).float().div(255))  # Convert back to tensor
        #trans.ToTensor()

])
    else:
        if args.transforms_mode == 'centerv2':
            img_train_transforms = trans.Compose([
                trans.RandomResizedCrop(
                    args.image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                trans.ToTensor()
            ])
            oct_train_transforms = trans.Compose([
                trans.ToTensor()
            ])
            img_val_transforms = trans.Compose([
                trans.CenterCrop((2000, 2000)),
                trans.Resize((args.image_size, args.image_size)),
                trans.ToTensor(),
            ])
            oct_val_transforms = trans.Compose([
            trans.Lambda(lambda x: torch.from_numpy(np.array(x)).float().div(255))  # Convert back to tensor
            ])

    train_dataset = GAMMA_sub1_dataset(dataset_root=args.train_root,
                                       img_transforms=img_train_transforms,
                                       oct_transforms=oct_train_transforms,
                                       filelists=train_filelists,
                                       label_file=args.label_file)

    val_dataset = GAMMA_sub1_dataset(dataset_root=args.train_root,
                                     img_transforms=img_val_transforms,
                                     oct_transforms=oct_val_transforms,
                                     filelists=val_filelists,
                                     label_file=args.label_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    return train_loader, val_loader


def main():
    print("Starting main function...")
    args = get_argparser().parse_args()

    train_loader, val_dataloader = get_dataloader(args)
    print("Data loaders created.")

    print("Initializing model...")
    if args.model_mode == '18':
        model = Dual_network18()
    elif args.model_mode == '34':
        model = Dual_RETFound(
            fundus_ckpt="C:/Users/kaust/Downloads/RETFound_mae_natureOCT.pth",
            oct_ckpt="C:/Users/kaust/Downloads/RETFound_mae_natureCFP.pth"
        )
        print("Model initialized.")
    elif args.model_mode == 'x50':
        model = Dual_networkx50()
    elif args.model_mode == 'b0':
        model = Dual_efficientnetb0()
    elif args.model_mode == 'b1':
        model = Dual_efficientnetb1()
    elif args.model_mode == 'b2':
        model = Dual_efficientnetb2()
    elif args.model_mode == 'b3':
        model = Dual_efficientnetb3()
    elif args.model_mode == 'b4':
        model = Dual_efficientnetb4()
    elif args.model_mode == 'b5':
        model = Dual_efficientnetb5()
    elif args.model_mode == 'b6':
        model = Dual_efficientnetb6()
    elif args.model_mode == 'b7':
        model = Dual_efficientnetb7()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model and optimizer initialized.")


    if args.loss_type == 'fl':
        criterion = FocalLoss(class_num=3, gamma=args.gamma)
    elif args.loss_type == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        print("Criterion initialized.")

    print("Starting training...")
    try:
        train(args, model, args.iters, train_loader, val_dataloader, optimizer, criterion, log_interval=50,
          eval_interval=50)
        print("Training completed.")
    
    except Exception as e:
        print(f"Error during training: {e}")



if __name__ == '__main__':
    main()