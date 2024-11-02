import random
import os
import pandas as pd
import sys
import torch
import numpy as np
import torch.nn.functional as F
from newmodel import create_model, BaseLoss, MaskLoss
from tqdm import tqdm
import data_preprocess.dataset as dataset
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from newmodel import BaseLoss
import argparse
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# model_weight_path = "./weights20220429/model-5.pth"
# mask_path = os.getcwd()+'/mask.npy'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data_loader, device, important=True, getMask=False):
    model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    base_loss_function = BaseLoss(device=device)
    accu_loss = torch.zeros(1).to(device)
    accu_num_pos = torch.zeros(1).to(device)
    sample_num = 0
    mask_all = np.zeros(914)
    data_loader = tqdm(data_loader)
    mask_l = []
    auc_labels =[]
    auc_preds =[]
    pred_class_l = []
    for step, data in enumerate(data_loader):
        exp, label = data
        exp = exp.float()
        # label = label.T
        label = label[:,0]
        sample_num += exp.shape[0]
        _, pred_1, mask = model(exp.to(device), important=important)
        # print(mask)

        # for i in mask:
        #     count = 0
        #     temp = []
        #     for j in range(len(i)):
        #         if 1 - i[j] <= 0.1:
        #             count += 1
        #             temp.append(j)
        #     np.save("newproject/2_new_record_batchsize16_3/_20240117/{}.npy".format(len(temp)),temp)
        #     exit()
        loss_pos = base_loss_function(pred_1, label)

        # total_loss_tmp.backward()

        accu_loss += loss_pos.detach()

        pred_classes_pos = torch.max(pred_1, dim=1)[1]
        accu_num_pos += torch.eq(pred_classes_pos, label.to(device)).sum()

        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred_1)

        pred_class_l.append(pred_classes_pos.cpu().numpy())
        auc_labels.append(label.cpu().numpy())
        auc_preds.append(pred_probs[:, 1].cpu().numpy())

        data_loader.desc = "[valid epoch {}] loss: {:.3f},  acc: {:.3f}".format(step,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num_pos.item() / sample_num
                                                                                )
        if mask is not None:
            for m in mask:
                mask_l.append(m)

    mask_all /= step
    # if getMask:
    #     np.save("mask_permute.npy", mask_all)
    label = np.concatenate(auc_labels, axis=0)
    pred_probs = np.concatenate(auc_preds, axis=0)
    pred_classes = np.concatenate(pred_class_l, axis=0)
    auc = roc_auc_score(label, pred_probs)
    precision = precision_score(label, pred_classes, average='binary')
    recall = recall_score(label, pred_classes, average='binary')
    f1 = f1_score(label, pred_classes, average='binary')
    print('auc:', round(auc, 4), 'precision:', round(precision, 4), 'recall:', round(recall, 4), 'f1:', round(f1, 4))
    return accu_loss.item() / (step + 1), accu_num_pos.item() / sample_num, mask_l, auc, precision, recall, f1



def prediect(x_test_path, y_test_path, num_classes, model_weight_path, ismask=True, is_all_transformer = False, is_all_mlp = False, important=True,
             laten=False, save_att='X_att', save_lantent='X_lat', n_step=10000, cutoff=0.1, batch_size=50, embed_dim=48,
             depth=2, num_heads=4, device="cuda:7"):
    set_seed(1)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    # _, _, _, _, x_test, y_test = dataset.Load2Dataset(adata)
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    num_genes = len(x_test[0])
    print(num_genes)
    # mask_path = os.getcwd()+project+'/mask.npy'
    # project_path = os.getcwd() + '/%s' % project
    # if os.path.exists(project_path) is False:
    #     os.makedirs(project_path)
    model = create_model(num_classes=num_classes, num_genes=num_genes, embed_dim=embed_dim, has_logits=False,
                         depth=depth, num_heads=num_heads, ismask=ismask, is_all_transformer = is_all_transformer, is_all_mlp = is_all_mlp).to(device)


    # load model weights
    # device_ids = [1, 2]
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", len(device_ids), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids)

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm = {}
    print(model_weight_path)
    for name, parameters in model.named_parameters():
        # print(name,':',parameters.size())
        parm[name] = parameters.detach().cpu().numpy()

    test_dataset = dataset.MyDataSet(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True, drop_last=False)
    with torch.no_grad():
        # predict class
        y_pred, y_test, mask_l, auc, precision, recall, f1 = evaluate(model=model,
                                  data_loader=data_loader,
                                  device=device,
                                  important=important)
        return y_pred, y_test, mask_l, auc, precision, recall, f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', default='train', type=str)
    parser.add_argument("--device", default='cuda:7', type=str)
    parser.add_argument("--x_test_path", default='data/0000_x_train.npy', type=str)
    parser.add_argument("--model_weight_path", default='newproject/base_0000/model-9.pth', type=str)
    parser.add_argument("--isresult", action='store_true', default=False)
    parser.add_argument("--use_argsort", action='store_true', default=False)
    parser.add_argument("--save_path", default='select_results/base_0000/', type=str)
    parser.add_argument("--ismask", action='store_true', default=False)
    parser.add_argument("--is_all_transformer", action='store_true', default=False)
    parser.add_argument("--is_all_mlp", action='store_true', default=False)
    parser.add_argument("--important_test", action='store_true', default=False)


    args = parser.parse_args()
    x_test_path = args.x_test_path
    y_test_path = x_test_path.replace('_x_', '_y_')
    x_train_path = x_test_path.replace('_test.npy', '_train.npy')
    y_train_path = x_train_path.replace('_x_', '_y_')
    result = args.isresult

    # x_test_path = '/opt/cr/metabolism/code/MoreData/2/x_train_attr.npy'
    # y_test_path = '/opt/cr/metabolism/code/MoreData/2/y_train_attr.npy'
    # x_test_path = 'select_results/test/p_x_test.npy'
    # y_test_path = 'MoreData/2/y_test_attr.npy'
    print('y_test_path', y_test_path)
    print('x_test_path', x_test_path)
    accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
        # x_test_path='/opt/cr/metabolism/code/MoreData/2/x_test_attr.npy',
        y_test_path=y_test_path,
        # x_test_path='/opt/cr/metabolism/code/MoreData/2/x_train_attr.npy',
        # y_test_path='/opt/cr/metabolism/code/MoreData/2/y_train_attr.npy',
        # x_test_path='select_results/test/p_x_test.npy',
        # y_test_path='MoreData/2/y_test_attr.npy',
        x_test_path=x_test_path,
                              num_classes=2,
                              batch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=4,
                              model_weight_path=args.model_weight_path,
                              ismask=args.ismask,
        is_all_transformer=args.is_all_transformer,
        is_all_mlp=args.is_all_mlp,
                              important=True,
                              device=args.device
                              )

    project = 'newproject/' + args.save_name
    project_path = os.getcwd()+'/%s'%project
    with open(os.path.join(project_path, 'result_test.txt'), 'w') as file:
        file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

    if args.important_test == True:
        x_train_path = x_test_path
        y_train_path = x_train_path.replace('_x_', '_y_')
        x_test_path = x_test_path.replace('_train.npy', '_test.npy')
        y_test_path = x_test_path.replace('_x_', '_y_')
        print('result_test_important')
        print('y_test_path', y_test_path)
        print('x_test_path', x_test_path)
        print('*' * 30)
        accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
            y_test_path=y_test_path,
            x_test_path=x_test_path,
            num_classes=2,
            batch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=4,
            model_weight_path=args.model_weight_path,
            ismask=args.ismask,
            is_all_transformer=args.is_all_transformer,
            is_all_mlp=args.is_all_mlp,
            important=True,
            device=args.device
        )

        project = 'newproject/' + args.save_name
        project_path = os.getcwd() + '/%s' % project
        with open(os.path.join(project_path, 'result_test_important.txt'), 'w') as file:
            file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

        accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
            y_test_path=y_test_path,
            x_test_path=x_test_path,
            num_classes=2,
            batch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=4,
            model_weight_path=args.model_weight_path,
            ismask=args.ismask,
            is_all_transformer=args.is_all_transformer,
            is_all_mlp=args.is_all_mlp,
            important=False,
            device=args.device
        )

        project = 'newproject/' + args.save_name
        project_path = os.getcwd() + '/%s' % project
        with open(os.path.join(project_path, 'result_test_unimportant.txt'), 'w') as file:
            file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

        print('result_train_important')
        print('y_train_path', y_train_path)
        print('x_train_path', x_train_path)
        print('*' * 30)
        accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
            y_test_path=y_train_path,
            x_test_path=x_train_path,
            num_classes=2,
            batch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=4,
            model_weight_path=args.model_weight_path,
            ismask=args.ismask,
            is_all_transformer=args.is_all_transformer,
            is_all_mlp=args.is_all_mlp,
            important=True,
            device=args.device
        )

        project = 'newproject/' + args.save_name
        project_path = os.getcwd() + '/%s' % project
        with open(os.path.join(project_path, 'result_train_important.txt'), 'w') as file:
            file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

        accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
            y_test_path=y_train_path,
            x_test_path=x_train_path,
            num_classes=2,
            batch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=4,
            model_weight_path=args.model_weight_path,
            ismask=args.ismask,
            is_all_transformer=args.is_all_transformer,
            is_all_mlp=args.is_all_mlp,
            important=False,
            device=args.device
        )

        project = 'newproject/' + args.save_name
        project_path = os.getcwd() + '/%s' % project
        with open(os.path.join(project_path, 'result_train_unimportant.txt'), 'w') as file:
            file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

        exit(0)


    if result == False:

        print(len(mask_l))
        # if len(mask_l) == 0:
        print('sparse', torch.mean(sum(mask_l)/len(mask_l)))
        mask_result = torch.stack(mask_l, dim=0)
        print('mask_result', mask_result.shape)
        bs = mask_result.shape[0] // 2
        c = mask_result.shape[1]
        # print('mask_result', mask_result.shape)
        mask_result = torch.sum(mask_result, dim=0)
        # print('mask_result', mask_result.shape)
        print(torch.max(mask_result), torch.min(mask_result))
        mask_result = mask_result.detach().cpu().numpy()

        if args.use_argsort == False:
            indices_positive = np.where(mask_result > bs)[0]
            indices_negtive = np.where(mask_result <= bs)[0]
        else:
            sorted_indices = np.argsort(mask_result)
            indices_positive = sorted_indices[-230:]
            indices_positive.sort()
            indices_negtive = sorted_indices[:-230]
            indices_negtive.sort()


        print('indices_positive', indices_positive)
        # print('indices_positive', indices_positive)
        print('len(indices_positive)', len(indices_positive))
        print('len(indices_negtive)', len(indices_negtive))
        #
        # # 读取属性名字的xlsx文件
        big_array_attributes_file = 'YQF-pos-name.xlsx'
        if c != 914:
            small_array_attributes_file =  os.path.join(x_test_path.split('/')[0], x_test_path.split('/')[1].split('_')[0]+ '_common_similar_attributes.xlsx').replace('data_all', 'data_select')
        else:
            small_array_attributes_file = 'YQF-pos-name.xlsx'

        # 读取属性名字
        big_array_attributes = pd.read_excel(big_array_attributes_file, header=None).iloc[1:, 0].tolist()
        small_array_attributes = pd.read_excel(small_array_attributes_file, header=None).iloc[1:, 0].tolist()

        print('big_array_attributes', big_array_attributes[:4])
        print('small_array_attributes', small_array_attributes[:4])
        small_array_attributes = [small_array_attributes[i] for i in indices_positive]

        print('small_array_attributes_1', len(small_array_attributes))

        # 确认这些选中的属性名字在大数组中，并获取它们的索引
        indices_to_remove = [big_array_attributes.index(attr) for attr in small_array_attributes if
                             attr in big_array_attributes]

        # 获取未被选择的索引
        indices_to_keep = [i for i in range(len(big_array_attributes)) if i not in indices_to_remove]
        indices_positive = indices_to_remove
        indices_negtive = indices_to_keep
        print('indices_to_remove', len(indices_to_remove))
        print('indices_to_keep', len(indices_to_keep))



        # exit(0)
        # # 打印要移除的索引（用于调试）
        # print("Indices to remove:", indices_to_remove)
        #
        # # 创建一个布尔掩码，表示哪些列需要保留
        # mask = np.ones(len(big_array_attributes), dtype=bool)
        # mask[indices_to_remove] = False
        #
        # # 使用掩码从大数组中移除选中的属性
        # filtered_big_array = big_array[:, mask]
        #
        # # 从大数组属性列表中移除选中的属性
        # filtered_big_array_attributes = [attr for i, attr in enumerate(big_array_attributes) if mask[i]]

        def dealwithPartialList(x_test_path, save_path, indices, save_name):
            x_train_path = x_test_path.replace('data_select', 'data_all')
            y_train_path = x_train_path.replace('_x_', '_y_').replace('data_select', 'data_all')
            x_test_path = x_test_path.replace('_train.npy', '_test.npy').replace('data_select', 'data_all')
            # x_test_path = x_test_path.replace('_train_', '_test_')
            y_test_path = x_test_path.replace('_x_', '_y_').replace('data_select', 'data_all')
            x_valid_path = x_train_path.replace('_train.npy', '_valid.npy')
            y_valid_path = x_valid_path.replace('_x_', '_y_')


            # x_train_path = x_test_path
            # y_train_path = x_train_path.replace('_x_', '_y_')
            # x_test_path = x_test_path.replace('_train.npy', '_test.npy')
            # # x_test_path = x_test_path.replace('_train_', '_test_')
            # y_test_path = x_test_path.replace('_x_', '_y_')
            # x_valid_path = x_train_path.replace('_train.npy', '_valid.npy')
            # y_valid_path = x_valid_path.replace('_x_', '_y_')


            # print('x_train_path', x_train_path)
            # print('x_valid_path', x_valid_path)
            # print('x_test_path', x_test_path)
            # print('y_train_path', y_train_path)
            # print('y_valid_path', y_valid_path)
            # print('y_test_path', y_test_path)
            # print('x_train_path', x_train_path)
            # print('x_test_path', x_test_path)

            print('x_train_path', x_train_path)
            print('x_valid_path', x_valid_path)
            print('x_test_path', x_test_path)
            print('y_train_path', y_train_path)
            print('y_valid_path', y_valid_path)
            print('y_test_path', y_test_path)

            x_train = np.load(x_train_path)
            x_valid = np.load(x_valid_path)
            x_test = np.load(x_test_path)
            y_train = np.load(y_train_path)
            y_valid = np.load(y_valid_path)
            y_test = np.load(y_test_path)
            # print(len(x_train))
            # print(len(x_valid))
            # print(len(x_test))
            # print('x_train', x_train.shape)
            # print('x_valid', x_valid.shape)
            # print('x_test', x_test.shape)
            x_train_new = x_train[:, indices]
            x_valid_new = x_valid[:, indices]
            x_test_new = x_test[:, indices]
            # print('x_train', x_train.shape)
            # print('x_train_new', x_train_new.shape)
            # print('x_valid_new', x_valid_new.shape)
            # print('x_test_new', x_test_new.shape)

            np.save(os.path.join(save_path, save_name +  '_x_train.npy'), x_train_new)
            np.save(os.path.join(save_path, save_name +  '_x_valid.npy'), x_valid_new)
            np.save(os.path.join(save_path, save_name +  '_x_test.npy'), x_test_new)


            np.save(os.path.join(save_path, save_name + '_y_train.npy'), y_train)
            np.save(os.path.join(save_path, save_name + '_y_valid.npy'), y_valid)
            np.save(os.path.join(save_path, save_name + '_y_test.npy'), y_test)


        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        dealwithPartialList(x_test_path, save_path,  indices_positive, 'p')
        dealwithPartialList(x_test_path, save_path, indices_negtive, 'n')

        with open(os.path.join(project_path, 'result_select_p.txt'), 'w') as file:
            # 在文件中写入要新增的行
            file.write(' '.join(map(str, indices_positive)) + '\n')  # 请将你要添加的内容替换为这里的文本
        with open(os.path.join(project_path, 'result_select_n.txt'), 'w') as file:
            # 在文件中写入要新增的行
            file.write(' '.join(map(str, indices_negtive)) + '\n')  # 请将你要添加的内容替换为这里的文本
    else:
        x_test_path = args.x_test_path
        y_test_path = x_test_path.replace('_x_', '_y_')
        x_train_path = x_test_path.replace('_test.npy', '_train.npy')
        y_train_path = x_train_path.replace('_x_', '_y_')
        accu_loss, accu_num_pos, mask_l, auc, precision, recall, f1 = prediect(
            # x_test_path='/opt/cr/metabolism/code/MoreData/2/x_test_attr.npy',
            y_test_path=y_train_path,
            # x_test_path='/opt/cr/metabolism/code/MoreData/2/x_train_attr.npy',
            # y_test_path='/opt/cr/metabolism/code/MoreData/2/y_train_attr.npy',
            # x_test_path='select_results/test/p_x_test.npy',
            # y_test_path='MoreData/2/y_test_attr.npy',
            x_test_path=x_train_path,
            num_classes=2,
            batch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=4,
            model_weight_path=args.model_weight_path,
            ismask=args.ismask,
            is_all_transformer=args.is_all_transformer,
            is_all_mlp=args.is_all_mlp,
            important=True,
            device=args.device
        )

        project = 'newproject/' + args.save_name
        project_path = os.getcwd() + '/%s' % project
        with open(os.path.join(project_path, 'result_train.txt'), 'w') as file:
            # 在文件中写入要新增的行
            file.write(str(accu_num_pos) + ' ' + str(auc) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(f1) + '\n')  # 请将你要添加的内容替换为这里的文本

