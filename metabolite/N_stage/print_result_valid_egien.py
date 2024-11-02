import os
import numpy as np
import pandas as pd

save_name_l = ['base_incre']
tgt_sparse_l = ['0.01', '0.016', '0.021', '0.04','0.055', '0.08', '0.2']
save_path_l = ['']
base_path = './newproject/'
columns = []
for i in range(6):
    columns.append(str(i).zfill(4))
columns.append('average')

base_names = ['']
for base_name_id, base_name in enumerate(base_names):

    for save_name_id, save_name in enumerate(save_name_l):
        results = []
        for tgt_sparse_id, tgt_sparse in enumerate(tgt_sparse_l):
            result_acc_l = []
            result_auc_l = []
            result_precision_l = []
            result_recall_l = []
            result_f1_l = []
            for i in range(6):
                subfile = save_name + '_' + base_name + str(i).zfill(4) + '_' + save_path_l[save_name_id] +'__' + tgt_sparse_l[tgt_sparse_id]
                print('*' * 40)
                print(subfile)
                print(os.path.join(base_path, subfile, 'result_test.txt'))
                with open(os.path.join(base_path, subfile, 'result_test.txt'), 'r') as file:
                    line = file.readline()
                # print(line)
                result_acc_l.append("{:.4f}".format(round(float(line.split(' ')[0]), 4)))
                result_auc_l.append("{:.4f}".format(round(float(line.split(' ')[1]), 4)))
                result_precision_l.append("{:.4f}".format(round(float(line.split(' ')[2]), 4)))
                result_recall_l.append("{:.4f}".format(round(float(line.split(' ')[3]), 4)))
                result_f1_l.append("{:.4f}".format(round(float(line.split(' ')[4]), 4)))


            result_acc_l = [float(num) for num in result_acc_l]
            result_acc_l.append(round(sum(result_acc_l) / len(result_acc_l), 4))

            result_auc_l = [float(num) for num in result_auc_l]
            result_auc_l.append(round(sum(result_auc_l) / len(result_auc_l), 4))

            result_precision_l = [float(num) for num in result_precision_l]
            result_precision_l.append(round(sum(result_precision_l) / len(result_precision_l), 4))

            result_recall_l = [float(num) for num in result_recall_l]
            result_recall_l.append(round(sum(result_recall_l) / len(result_recall_l), 4))

            result_f1_l = [float(num) for num in result_f1_l]
            result_f1_l.append(round(sum(result_f1_l) / len(result_f1_l), 4))

            print('result_f1_l', result_f1_l)

            results_tmp = []
            for auc, f1, acc, precision, recall in zip(result_auc_l, result_f1_l, result_acc_l, result_precision_l,
                                                       result_recall_l):
                results_tmp.append(f"{auc}/{f1}/{acc}/{precision}/{recall}")
            results.append(results_tmp)
        for r in results:
            print(r)
        r_tmp = ['model']
        for i in range(6):
            r_tmp.append(str(i).zfill(4))
        r_tmp.append('Average')
        results.insert(0, r_tmp)
        for tgt_sparse_idx, tgt_sparse in enumerate(tgt_sparse_l):
            results[tgt_sparse_idx+1].insert(0, str(tgt_sparse))
        # 将列表转换为 DataFrame
        df = pd.DataFrame(results)

        # 保存到 Excel 文件
        df.to_excel(os.path.join('newproject', 'incre_results'  + '.xlsx'), index=False)
            # print('columns', columns)
            # results = [
            #     f"{auc}/{f1}/{acc}/{precision}/{recall}"
            #     for auc, f1, acc, precision, recall in zip(result_auc_l, result_f1_l, result_acc_l, result_precision_l, result_recall_l)
            # ]
            # print('results', results)
            # df = pd.DataFrame([results], columns=columns)
            # print(save_name + '_' + tgt_sparse + '_select_results.xlsx')
            # df.to_excel( os.path.join('newproject', 'incre_results_' + tgt_sparse + '.xlsx'), index=False)

