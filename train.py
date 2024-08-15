import os
import tqdm
import time
import numpy as np
from tensorboardX import SummaryWriter
from utils import *
from evaluation import fast_hist, Evaluator
import torch

def val(args, model,  criterion,num_classes,dataloader_val, epoch, loss_train_mean, iou_train_mean, writer):
    print('Val...')
    start = time.time()

    with torch.no_grad():
        model.cuda()
        model.eval()
        loss_record = []
        hist = np.zeros((num_classes, num_classes))
        evaluator = Evaluator(num_classes)

        for i, (img,  label) in enumerate(dataloader_val):
            img, label = img.cuda(),  label.cuda()

            output= model(img)
            loss = criterion(output, label)
            predict = torch.argmax(output, 1)
            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()
            hist += fast_hist(pre, lab, num_classes)
            loss_record.append(loss.item())

        loss_val_mean = np.mean(loss_record)
        pa = evaluator.pixel_accuracy(hist)
        recall = evaluator.recall(hist)
        precision = evaluator.precision(hist)
        f1 = evaluator.f1_score(hist)
        miou = evaluator.mean_intersection_over_union(hist)
        fwiou = evaluator.frequency_weighted_intersection_over_union(hist)
        cpa = evaluator.cpa(hist)

        str_ = ("%15.5g;" * 15) % (epoch+1, loss_train_mean, loss_val_mean, pa, cpa[0], cpa[1], cpa[2], cpa[3], cpa[4],
                                   recall, precision, f1, iou_train_mean, miou, fwiou)

        with open(os.path.join(args.summary_path, args.dir_name, '')+args.dir_name+'_result.txt', 'a') as f:
            f.write(str_ + '\n')

        print('Val_loss:    {:}'.format(loss_val_mean))
        print('PA:          {:}'.format(pa))
        print('Recall:      {:}'.format(recall))
        print('Precision:   {:}'.format(precision))
        print('F1:          {:}'.format(f1))
        print('Miou:        {:}'.format(miou))
        print('FWiou:       {:}'.format(fwiou))
        print('Eval_time:   {:}s'.format(time.time() - start))

        writer.add_scalars('loss', {'train_loss': loss_train_mean, 'val_loss': loss_val_mean}, epoch+1)
        writer.add_scalars('miou', {'train_miou': iou_train_mean, 'val_miou': miou}, epoch+1)
        writer.add_scalar('{}_Loss'.format('val'), loss_val_mean, epoch+1)
        writer.add_scalar('{}_Pa'.format('val'), pa, epoch+1)
        writer.add_scalar('{}_Recall'.format('val'), recall, epoch+1)
        writer.add_scalar('{}_Precision'.format('val'), precision, epoch+1)
        writer.add_scalar('{}_F1'.format('val'), f1, epoch+1)
        writer.add_scalar('{}_Miou'.format('val'), miou, epoch+1)
        writer.add_scalar('{}_FWiou'.format('val'), fwiou, epoch + 1)
        return miou,hist


def train(args, model,  optimizer, criterion, dataloader_train, dataloader_val, exp_lr_scheduler, detail_aggregate_loss):
    print("Train...")
    miou_max = args.miou_max
    hist_max = np.zeros((args.num_classes, args.num_classes))
    miou_max_save_path = args.summary_path + args.dir_name
    miou_max_save_model_dict = None
    writer = SummaryWriter(logdir=os.path.join(args.summary_path, args.dir_name))
    s = ("%15s;" * 15) % ("epoch", "train_loss", "val_loss", "PA", "cloud", "cloud shadow", "snow/ice", "water", "land",
                          "Recall", "Precision", "F1", "Train_Miou", "Val_Miou", "FWiou")
    
    with open(os.path.join(args.summary_path, args.dir_name, '') + args.dir_name + '_result.txt', 'a') as f:
        f.write(s + '\n')

    for epoch in range(args.num_epochs):
        model.train()
        model.cuda()
        exp_lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr'] 
        tq = tqdm.tqdm(total=len(dataloader_train)*args.batch_size) 
        tq.set_description('epoch %d, lr %f' % (epoch+1, lr))
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        evaluator = Evaluator(args.num_classes)

        for i, (img, label) in enumerate(dataloader_train):
            img, label = img.cuda(), label.cuda()
            if args.warmup == 1 and epoch == 0:
                lr = args.lr / (len(dataloader_train) - i)
                tq.set_description('epoch %d, lr %f' % (epoch + 1, lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            output, detail_feat = model(img)
            predict = torch.argmax(output, 1)
            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()
            hist += fast_hist(pre, lab, args.num_classes)

            loss = criterion(output, label)
            bce_loss, dice_loss = detail_aggregate_loss(detail_feat, label)
            loss = loss + args.bcedice_factor * (bce_loss + dice_loss)
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

        iou_train_mean = evaluator.mean_intersection_over_union(hist)
        tq.close()
        loss_train_mean = np.mean(loss_record)
        print('Train Loss :{:.6f}'.format(loss_train_mean))
        print('Train MIoU :{:.6f}'.format(iou_train_mean))

        writer.add_scalar('{}_loss'.format('train'), loss_train_mean, epoch+1)
        writer.add_scalar('{}_miou'.format('train'), iou_train_mean, epoch+1)
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = args.save_model_path + 'epoch_{:}'.format(epoch)
            torch.save(model.state_dict(), save_path)

        if epoch % args.validation_step == 0:
            miou, val_hist = val(args,
                             model,
                             criterion,
                             args.num_classes,
                             dataloader_val,
                             epoch,
                             loss_train_mean,
                             iou_train_mean,
                             writer)
            if miou > miou_max:
                if not os.path.exists(args.summary_path + args.dir_name + '/checkpoints'):
                    os.makedirs(args.summary_path + args.dir_name + '/checkpoints')
                save_path = args.summary_path+args.dir_name+'/checkpoints/'+'miou_{:.6f}.pth'.format(miou)
                torch.save(model.state_dict(), save_path)
                miou_max = miou
                hist_max = val_hist
                miou_max_save_path = '{}{}/miou_{:.6f}_{:d}.pth'.format(args.summary_path, args.dir_name, miou_max,
                                                                        epoch)
                miou_max_save_model_dict = model.state_dict()
    writer.close()
    save_path = args.save_model_path + 'last.pth'
    if miou_max_save_model_dict is None:
        miou_max_save_path = '{}{}/last.pth'.format(args.summary_path, args.dir_name)
        miou_max_save_model_dict = model.state_dict()
    torch.save(miou_max_save_model_dict, miou_max_save_path)
    torch.save(model.state_dict(), save_path)
    np.savetxt(os.path.join(args.summary_path, args.dir_name, '') + args.dir_name + '_hist.txt', hist_max)
    
    return miou_max_save_path


