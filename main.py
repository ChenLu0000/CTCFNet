import os
import argparse
import torch
from torch import nn
from load_data import Seg
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from train import train
from PIL import Image
from torchvision import transforms
from lr_scheduler_poly import PolynomialLR
from utils import DetailAggregateLoss
from model.CTCFNet import CTCFNet

def main(model):
    parser = argparse.ArgumentParser('Seg')
    parser.add_argument('--num_epochs',             type=int,   default=200)
    parser.add_argument('--num_classes',            type=int,   default=5)
    parser.add_argument('--num_epoch_decay',        type=int,   default=70)
    parser.add_argument('--checkpoint_step',        type=int,   default=5)
    parser.add_argument('--validation_step',        type=int,   default=1)
    parser.add_argument('--batch_size',             type=int,   default=1)
    parser.add_argument('--num_workers',            type=int,   default=2)
    parser.add_argument('--lr',                     type=float, default=0.0001)
    parser.add_argument('--lr_scheduler',           type=int,   default=3)
    parser.add_argument('--lr_scheduler_gamma',     type=float, default=0.99)
    parser.add_argument('--warmup',                 type=int,   default=1)
    parser.add_argument('--warmup_num',             type=int,   default=1)
    parser.add_argument('--cuda',                   type=str,   default='0')
    parser.add_argument('--beta1',                  type=float, default=0.9)
    parser.add_argument('--beta2',                  type=float, default=0.999)
    parser.add_argument('--momentum',               type=float, default=0.9)
    parser.add_argument('--miou_max',               type=float, default=0.75)
    parser.add_argument('--dir_name',               type=str,   default='CTCFNet')
    parser.add_argument('--pretrained_model_path',  type=str,   default=None)
    parser.add_argument('--save_model_path',        type=str,   default="./checkpoints/")
    parser.add_argument('--data',                   type=str,   default='./datasets/name/')
    parser.add_argument('--summary_path',           type=str,   default='./summary/')
    parser.add_argument('--bcedice_factor',         type=float, default=1)
    args = parser.parse_args()

    tb = PrettyTable(['Num', 'Key', 'Value'])
    args_str = str(args)[10:-1].split(', ')

    for i, key_value in enumerate(args_str):
       key, value = key_value.split('=')[0], key_value.split('=')[1]
       tb.add_row([i + 1, key, value])
    print(tb)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.summary_path+args.dir_name):
        os.makedirs(args.summary_path+args.dir_name)

    train_path_img = os.path.join(args.data, 'train/images')
    train_path_label = os.path.join(args.data, 'train/labels')
    val_path_img = os.path.join(args.data, 'val/images')
    val_path_label = os.path.join(args.data, 'val/labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')

    train_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BILINEAR),
                                          transforms.ToTensor()
                                          ])
    dataset_train = Seg(
        train_path_img,
        train_path_label,
        csv_path,
        mode='train',
        transform=train_transform
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BILINEAR),
                                        transforms.ToTensor()
                                        ])
    dataset_val = Seg(
        val_path_img,
        val_path_label,
        csv_path,
        mode='val',
        transform=val_transform
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (args.beta1, args.beta2), weight_decay=1e-4)
    exp_lr_scheduler = PolynomialLR(optimizer, step_size=1, iter_max=args.num_epochs, power=2.0)
    criterion = nn.CrossEntropyLoss()
    detail_aggregate_loss = DetailAggregateLoss()

    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Done!')

    train(args,
          model,
          optimizer,
          criterion,
          dataloader_train,
          dataloader_val,
          exp_lr_scheduler,
          detail_aggregate_loss
          )

if __name__ == '__main__':
    model = CTCFNet(img_size=256,in_chans=3,class_dim=5,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=nn.LayerNorm, depths=[3, 3, 6, 3], sr_ratios=[8, 4, 2, 1])
    main(model)


