import random
import numpy as np
import sys
import pandas as pd
import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from torch.utils.tensorboard import SummaryWriter
import data_preprocess.dataset as dataset
from newmodel import create_model, BaseLoss, MaskLoss
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, optimizer_all, data_loader, device, epoch, all_epoches, tb_writer, args, tau):
    """
    Train the model and updata weights.
    """
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    base_loss_function = BaseLoss(device=device)
    mask_loss_function = MaskLoss()

    accu_loss = torch.zeros(1).to(device)
    base_loss = torch.zeros(1).to(device)
    mask_constrative_loss = torch.zeros(1).to(device)
    mask_constraint_loss = torch.zeros(1).to(device)
    mask_permute_loss = torch.zeros(1).to(device)
    # mask_loss = torch.zeros(1).to(device)
    accu_num_pos = torch.zeros(1).to(device)
    accu_num_neg = torch.zeros(1).to(device)
    invariant_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)
    sparse_loss_total = torch.zeros(1).to(device)
    class_loss_total = torch.zeros(1).to(device)
    invariant_loss_total = torch.zeros(1).to(device)
    sparse_print = torch.zeros(1).to(device)

    optimizer_all.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)

    count = 0
    # print('optimizer_all.param_groups[0]["lr"]', optimizer_all.param_groups[0]["lr"])
    for step, data in enumerate(data_loader):
        # feature_embed.MaskNet.fc1.weight
        # print('model.blocks.0.attn.qkv.weight', model.blocks[1].attn.qkv.weight)
        # print('model.feature_embed.MaskNet.fc1.weight', model.feature_embed.MaskNet.fc1.weight)
        optimizer_all.zero_grad()
        # print('step', step)
        exp, label = data
        exp = exp.float()
        # torch.autograd.set_detect_anomaly(True)
        # label = label.T
        label = label[:, 0]
        sample_num += exp.shape[0]
        _, pred_1, mask = model(exp.to(device), important=True, tau=tau, epoch=epoch)
        # _,pred_2,_ = model(exp.to(device),important = False)

        loss_pos = base_loss_function(pred_1, label)
        # loss_neg = base_loss_function(pred_2, label)
        base_loss_tmp = loss_pos
        # mask_loss_tmp, mask_constrastive_tmp, mask_constraint_tmp, permute_loss_tmp = mask_loss_function(loss_pos, loss_neg, mask, mask_permute, mask_)

        mask_mean = torch.mean(mask)
        # print('mask_mean', mask_mean)
        # sparse_loss = torch.relu(mask_mean - args.tgt_sparse)
        # sparse_loss = torch.relu(mask_mean-args.tgt_sparse) + torch.relu( (args.tgt_sparse-0.1) -mask_mean) * 10.
        # print('args.tgt_sparse', args.tgt_sparse)
        sparse_loss = torch.abs(mask_mean - args.tgt_sparse)
        # sparse_loss = torch.relu(mask_mean - args.tgt_sparse) + torch.relu(args.tgt_sparse - mask_mean) * 2.
        # sparse_loss = torch.abs( (mask_mean - args.tgt_sparse) * (mask_mean - args.tgt_sparse))
        # if epoch > 30:
        mask_loss = sparse_loss * args.sparse_loss_w
        # else:
        #     mask_loss = sparse_loss * 0.0
        # print('mask_loss', mask_loss)
        if args.use_class_loss == True:
            class_loss = 1 - torch.abs(2. * mask - 1).mean()
            mask_loss = mask_loss + class_loss * args.class_loss_w

        if args.use_invariant_loss == True:
            new_batch_size = args.batch_size // args.invariant_split_num * args.invariant_split_num
            new_mask = mask[:new_batch_size]
            new_masks = torch.chunk(new_mask, args.invariant_split_num, dim=0)
            invariant_loss_count = 0
            invariant_loss = 0
            for i in range(len(new_masks)):
                for j in range(i + 1, len(new_masks)):
                    invariant_loss = invariant_loss + torch.abs(new_masks[i] - new_masks[j]).sum()
                    invariant_loss_count = invariant_loss_count + 1
            invariant_loss = invariant_loss / invariant_loss_count
            mask_loss = mask_loss + invariant_loss * args.invariant_loss_w

        # if args.use_contrast_loss == True:

        total_loss_tmp = base_loss_tmp + mask_loss
        # total_loss_tmp = base_loss_tmp
        # total_loss_tmp = mask_loss
        total_loss_tmp.backward()
        # optimizer_mask.zero_grad()
        # mask_loss_tmp.backward(retain_graph=True)
        # base_loss_tmp.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        # optimizer_mask.step()
        optimizer_all.step()

        accu_loss += total_loss_tmp.detach()
        base_loss += base_loss_tmp.detach()
        # mask_constrative_loss += mask_constrastive_tmp.detach()
        # mask_constraint_loss += mask_constraint_tmp.detach()
        # mask_permute_loss += permute_loss_tmp.detach()

        pred_classes_pos = torch.max(pred_1, dim=1)[1]
        # pred_classes_neg = torch.max(pred_2, dim=1)[1]

        cur_acc = torch.eq(pred_classes_pos, label.to(device)).sum().item() / exp.shape[0]
        cur_loss = total_loss_tmp

        accu_num_pos += torch.eq(pred_classes_pos, label.to(device)).sum()
        # accu_num_neg += torch.eq(pred_classes_neg, label.to(device)).sum()
        sparse_print = sparse_print + torch.mean(mask)
        sparse_loss_total = sparse_loss_total + sparse_loss
        if args.use_class_loss == True:
            class_loss_total = class_loss_total + class_loss

        if args.use_invariant_loss == True:
            invariant_loss_total = invariant_loss_total + invariant_loss

        data_loader.desc = "[train epoch {}] loss: {:.3f}, " \
                           "base_loss: {:.3f}, " \
                           "acc_pos: {:.3f}, " \
                           "sparse:  {:.3f}, " \
                           "sparse_loss:  {:.3f}, " \
                           "lr:  {:.3f}, " \
            .format(
            epoch,
            accu_loss.item() / (step + 1),
            base_loss.item() / (step + 1),
            accu_num_pos.item() / sample_num,
            sparse_print.item() / (step + 1),
            sparse_loss_total.item() / (step + 1),
            optimizer_all.param_groups[0]['lr']
        )
        if not torch.isfinite(accu_loss):
            print('WARNING: non-finite loss, ending training ', accu_loss)
            sys.exit(1)
        count += exp.shape[0]
        # if count % (100 * exp.shape[0]) == 0:
        #     tb_writer.add_scalar("train_loss_", accu_loss.item() / (step + 1), (epoch + 1) * count)
        #     tb_writer.add_scalar("train_acc_", accu_num_pos.item() / sample_num, (epoch + 1) * count)
        #     tb_writer.add_scalar("acc_batch", cur_acc, (epoch + 1) * count)
        #     tb_writer.add_scalar("loss_batch", cur_loss, (epoch + 1) * count)
        #
        #     tb_writer.add_scalar("base_loss", base_loss.item() / (step + 1), (epoch + 1) * count)
        # tb_writer.add_scalar("mask_contrastive_loss", mask_constrative_loss.item() / (step + 1), (epoch + 1) * count)
        # tb_writer.add_scalar("mask_permute_loss", mask_permute_loss.item() / (step + 1), (epoch + 1) * count)
        # tb_writer.add_scalar("mask_constraint_loss", mask_constraint_loss.item() / (step + 1), (epoch + 1) * count)

    # mask_loss_function.lambda_mask *= 1.2

    return accu_loss.item() / (step + 1), accu_num_pos.item() / sample_num, base_loss.item() / (step + 1),


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, tb_writer, args):
    model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    base_loss_function = BaseLoss(device=device)
    mask_loss_function = MaskLoss()
    accu_loss = torch.zeros(1).to(device)
    base_loss = torch.zeros(1).to(device)
    mask_constrative_loss = torch.zeros(1).to(device)
    mask_constraint_loss = torch.zeros(1).to(device)
    mask_permute_loss = torch.zeros(1).to(device)
    accu_num_pos = torch.zeros(1).to(device)
    accu_num_neg = torch.zeros(1).to(device)
    sparse_loss_total = torch.zeros(1).to(device)
    class_loss_total = torch.zeros(1).to(device)
    invariant_loss_total = torch.zeros(1).to(device)
    sparse_print = torch.zeros(1).to(device)
    count = 0
    sample_num = 0
    acc_precision = 0
    acc_recall = 0
    acc_f1 = 0
    acc_auc = 0

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):

        exp, label = data
        exp = exp.float()

        # label = label.T
        label = label[:, 0]
        sample_num += exp.shape[0]
        _, pred_1, mask = model(exp.to(device), important=True)
        # _,pred_2,_ = model(exp.to(device),important = False)

        loss_pos = base_loss_function(pred_1, label)
        # loss_neg = base_loss_function(pred_2, label)
        base_loss_tmp = loss_pos
        # mask_loss_tmp, mask_constrastive_tmp, mask_constraint_tmp, permute_loss_tmp = mask_loss_function(loss_pos, loss_neg, mask, mask_permute,mask_)

        total_loss_tmp = base_loss_tmp
        # total_loss_tmp.backward()

        mask_mean = torch.mean(mask)
        # sparse_loss = torch.relu(mask_mean - args.tgt_sparse)
        # sparse_loss = torch.relu(mask_mean-args.tgt_sparse) + torch.relu( (args.tgt_sparse-0.1) -mask_mean)  * 10.
        sparse_loss = torch.abs(mask_mean - args.tgt_sparse)
        if args.use_class_loss == True:
            class_loss = 1 - torch.abs(2. * mask - 1).mean()
        if args.use_invariant_loss == True:
            new_batch_size = args.batch_size // args.invariant_split_num * args.invariant_split_num
            new_mask = mask[:new_batch_size]
            new_masks = torch.chunk(new_mask, args.invariant_split_num, dim=0)
            invariant_loss_count = 0
            invariant_loss = 0
            for i in range(len(new_masks)):
                for j in range(i + 1, len(new_masks)):
                    invariant_loss = invariant_loss + torch.abs(new_masks[i] - new_masks[j]).sum()
                    invariant_loss_count = invariant_loss_count + 1
            invariant_loss = invariant_loss / invariant_loss_count

        accu_loss += total_loss_tmp.detach()
        base_loss += base_loss_tmp.detach()
        # mask_constrative_loss += mask_constrastive_tmp.detach()
        # mask_constraint_loss += mask_constraint_tmp.detach()
        # mask_permute_loss += permute_loss_tmp.detach()
        sparse_print = sparse_print + torch.mean(mask)
        sparse_loss_total = sparse_loss_total + sparse_loss
        if args.use_class_loss == True:
            class_loss_total = class_loss_total + class_loss

        if args.use_invariant_loss == True:
            invariant_loss_total = invariant_loss_total + invariant_loss

        pred_classes_pos = torch.max(pred_1, dim=1)[1]

        cur_acc = torch.eq(pred_classes_pos, label.to(device)).sum().item() / exp.shape[0]
        cur_loss = total_loss_tmp

        accu_num_pos += torch.eq(pred_classes_pos, label.to(device)).sum()

        # # 计算 Precision, Recall, F1-Score
        # precision = precision_score(label.cpu().numpy(), pred_classes_pos.cpu().numpy(), average='binary')
        # recall = recall_score(label.cpu().numpy(), pred_classes_pos.cpu().numpy(), average='binary')
        # f1 = f1_score(label.cpu().numpy(), pred_classes_pos.cpu().numpy(), average='binary')
        # softmax = nn.Softmax(dim=1)
        # pred_probs = softmax(pred_1)
        # auc = roc_auc_score(label.cpu().numpy(), pred_probs[:, 1].cpu().numpy())
        #
        # acc_precision += precision
        # acc_recall += recall
        # acc_f1 += f1
        # acc_auc += auc

        # accu_num_neg += torch.eq(pred_classes_neg, label.to(device)).sum()
        data_loader.desc = ("[train epoch {}] "
                            "loss: {:.3f}, " \
                            "base_loss: {:.3f}, " \
                            "acc_pos: {:.3f}, " \
                            "sparse:  {:.3f}, " \
                            # "acc_precision: {:.3f}, " \
                            # "acc_recall: {:.3f}, " \
                            # "f1: {:.3f}, " \
                            # "auc: {:.3f}, " \

                            # "sparse_loss:  {:.3f}, " \
                            # "class_loss:  {:.3f}, " \
                            # "invariant_loss:  {:.3f}, " \
            .format(
            epoch,
            accu_loss.item() / (step + 1),
            base_loss.item() / (step + 1),
            accu_num_pos.item() / sample_num,
            sparse_print.item() / (step + 1),
            # acc_precision.item() / sample_num,
            # acc_recall.item() / sample_num,
            # acc_f1.item() / sample_num,
            # acc_auc.item() / sample_num,


            # sparse_loss_total.item() / (step + 1),
            # class_loss_total.item() / (step + 1),
            # invariant_loss_total.item() / (step + 1)
        ))

        count += exp.shape[0]
        if count % (100 * exp.shape[0]) == 0:
            tb_writer.add_scalar("valid_loss_", accu_loss.item() / (step + 1), (epoch + 1) * count)
            tb_writer.add_scalar("valid_acc_", accu_num_pos.item() / sample_num, (epoch + 1) * count)

            tb_writer.add_scalar("valid_acc_batch", cur_acc, count)
            tb_writer.add_scalar("valid_loss_batch", cur_loss, count)

            tb_writer.add_scalar("valid_base_loss", base_loss.item() / (step + 1), (epoch + 1) * count)
            # tb_writer.add_scalar("valid_mask_contrastive_loss", mask_constrative_loss.item() / (step + 1), (epoch + 1) * count)
            # tb_writer.add_scalar("valid_mask_permute_loss", mask_permute_loss.item() / (step + 1), (epoch + 1) * count)
            # tb_writer.add_scalar("valid_mask_constraint_loss", mask_constraint_loss.item() / (step + 1), (epoch + 1) * count)

    # return accu_loss.item() / (step + 1), accu_num_pos.item() / sample_num, acc_auc / sample_num
    return accu_loss.item() / (step + 1), accu_num_pos.item() / sample_num, 0


def train_one_epoch_without_mask(model, optimizer, data_loader, device, epoch, tb_writer):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    base_loss_function = BaseLoss(device=device)

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    count = 0
    for step, data in enumerate(data_loader):

        exp, label = data

        # label = label.T
        label = label[:, 0]
        sample_num += exp.shape[0]
        _, pred, mask = model(exp.to(device))

        loss = base_loss_function(pred, label)

        # total_loss_tmp.backward()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accu_loss += loss.detach()

        pred_classes = torch.max(pred, dim=1)[1]
        # print('pred_classes',pred_classes)
        # print('label',label)
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num,
                                                                               )
        count += exp.shape[0]
        if count % (100 * exp.shape[0]) == 0:
            tb_writer.add_scalar("train_loss_", accu_loss.item() / (step + 1), (epoch + 1) * count)
            tb_writer.add_scalar("train_acc_", accu_num.item() / sample_num, (epoch + 1) * count)

        if not torch.isfinite(accu_loss):
            print('WARNING: non-finite loss, ending training ', accu_loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_without_mask(model, data_loader, device, epoch, tb_writer):
    model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    base_loss_function = BaseLoss(device=device)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    count = 0
    acc_precision = 0
    acc_recall = 0
    acc_f1 = 0
    acc_auc = 0
    auc_labels = []
    auc_preds = []
    pred_class_l = []

    for step, data in enumerate(data_loader):

        exp, label = data

        # label = label.T
        label = label[:,0]
        sample_num += exp.shape[0]
        # print('here')
        # print('exp', exp.shape)
        _, pred, mask = model(exp.to(device))
        loss = base_loss_function(pred, label)

        # total_loss_tmp.backward()

        accu_loss += loss.detach()

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()

        # 计算 Precision, Recall, F1-Score
        # precision = precision_score(label.cpu().numpy(), pred_classes.cpu().numpy(), average='binary')
        # recall = recall_score(label.cpu().numpy(), pred_classes.cpu().numpy(), average='binary')
        # f1 = f1_score(label.cpu().numpy(), pred_classes.cpu().numpy(), average='binary')
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)

        pred_class_l.append(pred_classes.cpu().numpy())
        auc_labels.append(label.cpu().numpy())
        auc_preds.append(pred_probs[:, 1].cpu().numpy())

        # acc_precision += precision
        # acc_recall += recall
        # acc_f1 += f1
        # acc_auc += auc

        data_loader.desc = ("[valid epoch {}] "
                            "loss: {:.3f}, "
                            "acc: {:.3f} "
                            # "precision: {:.3f} "
                            # "recall: {:.3f} "
                            # "f1: {:.3f} "
                            # "auc: {:.3f} "
                            .
                            format(epoch,
                                   accu_loss.item() / (step + 1),
                                   accu_num.item() / sample_num,
                                   # acc_precision.item() / sample_num,
                                   # acc_recall.item() / sample_num,
                                   # acc_f1.item() / sample_num,
                                   # acc_auc.item() / sample_num,
                                   ))
        count += exp.shape[0]
        if count % (100 * exp.shape[0]) == 0:
            tb_writer.add_scalar("valid_loss_", accu_loss.item() / (step + 1), (epoch + 1) * count)
            tb_writer.add_scalar("valid_acc_", accu_num.item() / sample_num, (epoch + 1) * count)

    label = np.concatenate(auc_labels, axis=0)
    pred_probs = np.concatenate(auc_preds, axis=0)
    pred_classes = np.concatenate(pred_class_l, axis=0)
    auc = roc_auc_score(label, pred_probs)
    precision = precision_score(label, pred_classes, average='binary')
    recall = recall_score(label, pred_classes, average='binary')
    f1 = f1_score(label, pred_classes, average='binary')
    print('auc:', round(auc, 4), 'precision:', round(precision, 4), 'recall:', round(recall, 4), 'f1:', round(f1, 4))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, recall, f1


def fit_model(x_train_path, y_train_path, x_valid_path, y_valid_path, num_classes, device, type_path, batch_size=8,
              embed_dim=48, depth=2, num_heads=4, ismask=False, is_all_transformer=False, is_all_mlp=False,
              pre_weights='', epochs=10, lrf=0.001, lr=0.0001, args=None):
    set_seed(1)

    x_train = np.load(x_train_path)
    print('x_train', x_train.shape)
    bs = x_train.shape[0]
    x_train = x_train
    y_train = np.load(y_train_path)
    x_valid = np.load(x_valid_path)
    y_valid = np.load(y_valid_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    # project = 'newproject/' + type_path+ '_%s'%today
    project = 'newproject/' + type_path
    project_path = os.getcwd() + '/%s' % project
    print("Make project path!")
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    # x_train, y_train, x_valid, y_valid, _, _ = dataset.Load2Dataset(data)
    # x_train = np.load(data_path +'/x_train_norm.npy')
    # y_train = np.load(data_path +'/y_train.npy')
    # x_valid = np.load(data_path +'/x_valid_norm.npy')
    # y_valid = np.load(data_path +'/y_valid.npy')
    train_dataset = dataset.MyDataSet(x_train, y_train)
    valid_dataset = dataset.MyDataSet(x_valid, y_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True, drop_last=False)
    if ismask == True:
        model = create_model(num_classes=num_classes,
                             num_genes=len(x_train[0]),
                             embed_dim=embed_dim,
                             depth=depth,
                             num_heads=num_heads,
                             ismask=ismask,
                             is_all_transformer=is_all_transformer,
                             is_all_mlp=is_all_mlp,
                             has_logits=False).to(device=device)

        # device_ids = [1, 2]
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", len(device_ids), "GPUs!")
        #     model = torch.nn.DataParallel(model, device_ids)

        if pre_weights != "":
            assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
            preweights_dict = torch.load(pre_weights, map_location=device)
            print(model.load_state_dict(preweights_dict, strict=False))
        print('Model builded!')
        print(type_path)

        # 冻结梯度

        # model.feature_embed.MaskNet.fc1.weight.requires_grad_ = False
        # model.feature_embed.MaskNet.fc1.bias.requires_grad_ = False
        # model.feature_embed.MaskNet.fc2.weight.requires_grad_ = False
        # model.feature_embed.MaskNet.fc2.bias.requires_grad_ = False
        # model.feature_embed.MaskNet.fc3.weight.requires_grad_ = False
        # model.feature_embed.MaskNet.fc3.bias.requires_grad_ = False
        # # 获取仍然需要梯度计算的参数
        # pg_all = [p for p in model.parameters() if p.requires_grad_]

        # model.feature_embed.MaskNet.fc1.weight.requires_grad_ = True
        # model.feature_embed.MaskNet.fc1.bias.requires_grad_ = True
        # model.feature_embed.MaskNet.fc2.weight.requires_grad_ = True
        # model.feature_embed.MaskNet.fc2.bias.requires_grad_ = True
        # model.feature_embed.MaskNet.fc3.weight.requires_grad_ = True
        # model.feature_embed.MaskNet.fc3.bias.requires_grad_ = True
        # print(model.feature_embed.MaskNet.fc3.bias.requires_grad_)
        # exit(0)
        # pg_mask = model.feature_embed.parameters()
        for name, param in model.named_parameters():
            print(f"Layer: {name}, Requires Grad: {param.requires_grad}")

        # optimizer_all = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        optimizer_all = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # optimizer_mask = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        scheduler_all = lr_scheduler.LambdaLR(optimizer_all, lr_lambda=lf)
        # scheduler_mask = lr_scheduler.LambdaLR(optimizer_mask, lr_lambda=lf)
        save_txt = open(os.path.join(project_path, 'result.txt'), 'w')
        train_acc_best = 0
        val_acc_best = 0
        val_acc_best_tau = 0

        initial_temperature = 0.4
        min_temperature = 0.0001
        decay_rate = 0.4
        for epoch in range(epochs):
            # tau = max(0.4 - ((epoch) / epochs), 0.0001)
            # tau = max(min_temperature, initial_temperature * (decay_rate ** epoch))
            tau = max(min_temperature, initial_temperature - (epoch / epochs) * (initial_temperature - min_temperature))
            # tau = max(0.004 - ((epoch) / epochs), 0.0001)
            # tau = 0.0001
            # tau = max(1 - ((epoch) / epochs), 0.4)
            train_loss, train_acc, base_loss = train_one_epoch(model=model,
                                                               optimizer_all=optimizer_all,
                                                               # optimizer_mask=optimizer_mask,
                                                               data_loader=train_loader,
                                                               device=device,
                                                               epoch=epoch,
                                                               tb_writer=tb_writer,
                                                               all_epoches=epochs,
                                                               args=args, tau=tau)
            scheduler_all.step()
            # scheduler_mask.step()

            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    val_loss, val_acc, val_auc = evaluate(model=model,
                                                 data_loader=valid_loader,
                                                 device=device,
                                                 epoch=epoch,
                                                 tb_writer=tb_writer,
                                                 args=args)
                tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
                tb_writer.add_scalar(tags[0], train_loss, epoch)
                tb_writer.add_scalar(tags[1], train_acc, epoch)
                tb_writer.add_scalar(tags[2], val_loss, epoch)
                tb_writer.add_scalar(tags[3], val_acc, epoch)
                tb_writer.add_scalar(tags[4], optimizer_all.param_groups[0]["lr"], epoch)
                tb_writer.add_scalar("base_loss_epoch", base_loss, epoch)

                torch.save(model.state_dict(), "/%s" % project_path + "/model-{}.pth".format(epoch))

                save_txt.write(' '.join([str(epoch), str(train_loss), str(train_acc), str(val_loss),
                                         str(val_acc)]) + '\n')  # 请将你要添加的内容替换为这里的文本

                if train_acc >= train_acc_best:
                    train_acc_best = train_acc
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-train-acc-best.pth")
                if val_acc >= val_acc_best:
                    val_acc_best = val_acc
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-val-acc-best.pth")
                if val_acc >= val_acc_best_tau and tau < 0.1:
                    val_acc_best_tau = val_acc
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-val-acc-best-tau.pth")

        save_txt.close()
        with open(os.path.join(project_path, 'result_best.txt'), 'w') as file:
            # 在文件中写入要新增的行
            file.write(' '.join([str(train_acc_best), str(val_acc_best)]) + '\n')  # 请将你要添加的内容替换为这里的文本
        print('Training finished!')
    else:
        model = create_model(num_classes=num_classes,
                             num_genes=len(x_train[0]),
                             embed_dim=embed_dim,
                             depth=depth,
                             num_heads=num_heads,
                             ismask=ismask,
                             is_all_transformer=is_all_transformer,
                             is_all_mlp=is_all_mlp,
                             has_logits=False).to(device=device)

        if pre_weights != "":
            assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
            preweights_dict = torch.load(pre_weights, map_location=device)
            print(model.load_state_dict(preweights_dict, strict=False))
        print('Model builded!')
        print(type_path)

        pg_all = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(pg_all, lr=lr, momentum=0.9, weight_decay=5e-5)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        save_txt = open(os.path.join(project_path, 'result.txt'), 'w')
        train_acc_best = 0
        val_acc_best = 0
        val_auc_best = 0
        val_precision_best = 0
        val_recall_best = 0
        val_f1_best = 0
        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch_without_mask(model=model,
                                                                 optimizer=optimizer,
                                                                 data_loader=train_loader,
                                                                 device=device,
                                                                 epoch=epoch,
                                                                 tb_writer=tb_writer)
            scheduler.step()
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_without_mask(model=model,
                                                                                                          data_loader=valid_loader,
                                                                                                          device=device,
                                                                                                          epoch=epoch,
                                                                                                          tb_writer=tb_writer)
                tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
                tb_writer.add_scalar(tags[0], train_loss, epoch)
                tb_writer.add_scalar(tags[1], train_acc, epoch)
                tb_writer.add_scalar(tags[2], val_loss, epoch)
                tb_writer.add_scalar(tags[3], val_acc, epoch)
                tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

                # torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))

                save_txt.write(' '.join(
                    [str(epoch), str(train_loss), str(train_acc), str(val_loss), str(val_acc),
                     str(val_auc)]) + '\n')  # 请将你要添加的内容替换为这里的文本

                if train_acc >= train_acc_best:
                    train_acc_best = train_acc
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-train-acc-best.pth")
                if val_acc >= val_acc_best:
                    val_acc_best = val_acc
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-val-acc-best.pth")
                if val_auc >= val_auc_best:
                    val_auc_best = val_auc
                    val_precision_best = val_precision
                    val_recall_best = val_recall
                    val_f1_best = val_f1
                    torch.save(model.state_dict(), "/%s" % project_path + "/model-val-auc-best.pth")

        save_txt.close()
        with open(os.path.join(project_path, 'result_best.txt'), 'w') as file:
            # 在文件中写入要新增的行
            file.write(' '.join([str(train_acc_best), str(val_acc_best), str(val_auc_best), str(val_precision_best),
                                 str(val_recall_best), str(val_f1_best)]) + '\n')  # 请将你要添加的内容替换为这里的文本

        print('Training finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', default='train', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--tgt_sparse', default=0.4, type=float)
    parser.add_argument("--use_class_loss", action='store_true', default=False)
    parser.add_argument("--use_invariant_loss", action='store_true', default=False)
    parser.add_argument('--invariant_split_num', default=3, type=int)
    parser.add_argument("--use_contrast_loss", action='store_true', default=False)
    parser.add_argument('--sparse_loss_w', default=3.0, type=float)
    parser.add_argument('--class_loss_w', default=2.0, type=float)
    parser.add_argument('--invariant_loss_w', default=0.0001, type=float)
    parser.add_argument('--contrast_loss_w', default=0.001, type=float)
    parser.add_argument("--ismask", action='store_true', default=False)
    parser.add_argument("--is_all_transformer", action='store_true', default=False)
    parser.add_argument("--is_all_mlp", action='store_true', default=False)
    parser.add_argument("--device", default='cuda:7', type=str)
    parser.add_argument("--x_train_path", default='data/0000_x_train.npy', type=str)
    parser.add_argument("--isresult", action='store_true', default=False)

    args = parser.parse_args()
    x_train_path = args.x_train_path
    y_train_path = x_train_path.replace('_x_', '_y_')
    if args.isresult == False:
        x_valid_path = x_train_path.replace('_train.npy', '_valid.npy')
        y_valid_path = x_valid_path.replace('_x_', '_y_')
    else:
        x_valid_path = x_train_path.replace('_train.npy', '_test.npy')
        y_valid_path = x_valid_path.replace('_x_', '_y_')
    print('y_valid_path', y_valid_path)
    fit_model(device=args.device,
              # x_train_path='MoreData/2/x_train_attr.npy',
              # x_train_path='select_results/test/p_x_train.npy',
              y_train_path=y_train_path,
              # x_valid_path='MoreData/2/x_valid_attr.npy',
              # x_valid_path='select_results/test/p_x_test.npy',
              y_valid_path=y_valid_path,
              x_train_path=x_train_path,
              # y_train_path='MoreData/2/y_train_attr.npy',
              x_valid_path=x_valid_path,
              # y_valid_path='MoreData/2/y_test_attr.npy',
              type_path=args.save_name,
              num_classes=2,
              batch_size=args.batch_size,
              embed_dim=192,
              depth=4,
              num_heads=4,
              # embed_dim=192,
              # depth=2,
              # num_heads=4,
              epochs=args.epochs,
              ismask=args.ismask,
              is_all_transformer=args.is_all_transformer,
              is_all_mlp=args.is_all_mlp,
              args=args
              )