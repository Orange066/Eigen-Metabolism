import os
import numpy as np
import pandas as pd


def metric_calculate(result_all, base_path, subfile_l,  select, txt_name,):
    result_acc_l = []
    result_auc_l = []
    result_precision_l = []
    result_recall_l = []
    result_f1_l = []
    result_all_curr = []

    for i in range(len(subfile_l)):
        subfile = subfile_l[i]
        # positive
        with open(os.path.join(base_path, subfile + select, txt_name), 'r') as file:
            line = file.readline()
        # print(line)
        result_acc_l.append("{:.4f}".format(round(float(line.split(' ')[0]), 2)))
        result_auc_l.append("{:.4f}".format(round(float(line.split(' ')[1]), 2)))
        result_precision_l.append("{:.4f}".format(round(float(line.split(' ')[2]), 2)))
        result_recall_l.append("{:.4f}".format(round(float(line.split(' ')[3]), 2)))
        result_f1_l.append("{:.4f}".format(round(float(line.split(' ')[4]), 2)))

        result_all_curr.append([result_auc_l[i], result_f1_l[i], result_acc_l[i], result_recall_l[i], result_precision_l[i]])

    result_all_tmp = []
    # print(result_auc_l)

    result_acc_l = [float(num) for num in result_acc_l]
    result_auc_l = [float(num) for num in result_auc_l]
    result_precision_l = [float(num) for num in result_precision_l]
    result_recall_l = [float(num) for num in result_recall_l]
    result_f1_l = [float(num) for num in result_f1_l]

    result_all_tmp.append("{:.4f}".format(sum(result_auc_l) / len(result_auc_l)))
    result_all_tmp.append("{:.4f}".format(sum(result_f1_l) / len(result_f1_l)))
    result_all_tmp.append("{:.4f}".format(sum(result_acc_l) / len(result_acc_l)))
    result_all_tmp.append("{:.4f}".format(sum(result_precision_l) / len(result_precision_l)))
    result_all_tmp.append("{:.4f}".format(sum(result_recall_l) / len(result_recall_l)))
    result_all_curr.append(result_all_tmp)
    # print(result_all_curr[-1])
    result_all.append(result_all_curr)
    return result_all

save_name_l = ['base_select']
tgt_sparse_l = ['0.01', '0.05', '0.10', '0.20', '0.40', '0.60', '0.80']
save_path_l = ['']
save_path_l = []
for i in range(2):
    save_path_l.append('')


base_path = './newproject/'
# percentage=0.01
# n = int(percentage * 914 + 0.5)  # 要选取的属性数量
percentage_l=[0.002, 0.01, 0.021, 0.023, 0.04, 0.06, 0.10]
n_l = []
for i in range(len(percentage_l)):
    n_l.append(int(percentage_l[i] * 914 + 0.5))


txt_name_l =[]
for save_name_id, save_name in enumerate(save_name_l):
    columns = []
    for i in range(2):
        columns.append(str(i).zfill(4))
    columns.append('average')

    xlsx_file_path = 'YQF-pos-name.xlsx'
    df = pd.read_excel(xlsx_file_path)
    column_five_values = df.iloc[:, 0].tolist()
    column_five_values = np.array(column_five_values)

    xlsx_file_path = 'YQF-pos-id.xlsx'
    df = pd.read_excel(xlsx_file_path)
    column_six_values = df.iloc[:, 0].tolist()
    column_six_values = np.array(column_six_values)

    xlsx_file_path = 'YQF-pos-kegg.xlsx'
    df = pd.read_excel(xlsx_file_path)
    # df = df.fillna(str(-1))
    column_ten_values = df.iloc[:, 0].tolist()
    column_ten_values = np.array(column_ten_values)

    print('here')
    for tgt_sparse_id, tgt_sparse in enumerate(tgt_sparse_l):
        # save_txt = open(os.path.join('newproject', save_name + '_' + tgt_sparse + '.txt'), 'w')
        # result_train_p_l = []
        # result_test_p_l = []
        # result_train_n_l = []
        # result_test_n_l = []
        # result_test_important_l = []
        # result_test_unimportant_l = []
        # result_train_important_l = []
        # result_train_unimportant_l = []
        results = []
        for i in range(914):
            results.append(0)
        subfile_l = []

        for i in range(2):
            # subfile = 'base_'+ str(i).zfill(4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id]
            subfile = save_name + '_' + str(i).zfill(4) + '_' + save_path_l[save_name_id] + '_' + tgt_sparse_l[tgt_sparse_id]
            subfile_l.append(subfile)
            # print('*' * 40)
            # print(subfile)
            with open(os.path.join(base_path, subfile, 'result_select_p.txt'), 'r') as file:
                line = file.readline()
            select_p = line.split(' ')
            for p in select_p:
                results[int(p)] += 1

        result_all = []
        result_all = metric_calculate(result_all, base_path, subfile_l, '_p', 'result_train.txt')
        result_all = metric_calculate(result_all, base_path, subfile_l, '_p', 'result_test.txt')
        result_all = metric_calculate(result_all, base_path, subfile_l, '_n', 'result_train.txt')
        result_all = metric_calculate(result_all, base_path, subfile_l, '_n', 'result_test.txt')

        results = np.array(results)
        # 参数

        # 获取按出现次数排序的索引
        sorted_indices = np.argsort(-results)

        # 选择前 n 个属性的索引
        selected_indices = sorted_indices[:n_l[tgt_sparse_id]]
        # print('selected_indices', selected_indices)

        # 获取最小出现次数
        min_count = results[selected_indices].min()
        # print('min_count', min_count)

        # 找到最小出现次数的所有索引并添加到选择集中
        min_count_indices = np.where(results == min_count)[0]
        selected_indices = np.union1d(selected_indices, min_count_indices)

        # 根据索引获取名字和出现次数
        selected_column_five_values = column_five_values[selected_indices]
        selected_column_six_values = column_six_values[selected_indices]
        selected_column_ten_values = column_ten_values[selected_indices]
        selected_results = results[selected_indices]
        # print('selected_indices_0', selected_indices)

        sorted_order = np.argsort(-selected_results)
        selected_column_five_values = selected_column_five_values[sorted_order]
        selected_column_six_values = selected_column_six_values[sorted_order]
        selected_column_ten_values = selected_column_ten_values[sorted_order]
        selected_results = selected_results[sorted_order]
        selected_indices = selected_indices[sorted_order]
        # print('selected_indices_0', selected_indices)

        # 保存到txt文件
        with open( os.path.join('newproject', 'selected_attributes_' +  str(percentage_l[tgt_sparse_id]) + '.txt') , 'w') as file:
            file.write('count: ' + str(len(selected_indices)) + '\n')
            for i in range(len(selected_indices)):
                # print('selected_indices[i]', selected_indices[i], type(selected_indices[i]))
                # print('selected_results[i]', selected_results[i], type(selected_results[i]))
                # print('selected_column_five_values[i]', selected_column_five_values[i], type(selected_column_five_values[i]))
                # print('selected_column_six_values[i]', selected_column_six_values[i], type(selected_column_six_values[i]))
                # print('selected_column_ten_values[i]', selected_column_ten_values[i], type(selected_column_ten_values[i]))
                file.write(str(selected_indices[i]) + ' ' + str(selected_results[i]) + ' ' + selected_column_five_values[i] +' ' + selected_column_six_values[i] +' ' + selected_column_ten_values[i] + '\n')
        print("文件已保存为 " + os.path.join('newproject', 'selected_attributes_' +  str(percentage_l[tgt_sparse_id]) + '.txt'))
        txt_name_l.append(os.path.join('newproject', 'selected_attributes_' +  str(percentage_l[tgt_sparse_id]) + '.txt'))

        with open(os.path.join('newproject', 'selected_attributes_all_' +  str(percentage_l[tgt_sparse_id]) + '.txt') , 'w') as file:
            for idx in range(len(sorted_indices)):
                i = sorted_indices[idx]
                file.write(
                    str(i) + ' ' + str(results[i]) + ' ' + column_five_values[
                        i] + ' ' + column_six_values[i] + ' ' + column_ten_values[i] + '\n')
        print("文件已保存为 'selected_attributes_all.txt'")

        r_l = []
        for r_sub in result_all:
            r_tmp = []
            for r in r_sub:
                r_tmp.append('/'.join(r))
            r_l.append(r_tmp)
        r_tmp = ['model']
        for i in range(2):
            r_tmp.append(str(i).zfill(4))
        r_tmp.append('Average')
        r_l.insert(0,r_tmp)
        r_l[1].insert(0,'select_valid')
        r_l[2].insert(0,'select_test')
        r_l[3].insert(0,'no_select_valid')
        r_l[4].insert(0,'no_select_test')
        # for r in r_l:
        #     print(r)

        # 将列表转换为 DataFrame
        df = pd.DataFrame(r_l)

        # 保存到 Excel 文件
        df.to_excel(os.path.join('newproject', 'selected_results_' + tgt_sparse + '.xlsx'), index=False)




        # exit(0)

select_l = []
# count_ =0
# print(txt_name_l)
# exit(0)
txt_incre_name_l = []
count_all_past = 0
for txt_name_idx, txt_name in enumerate(txt_name_l):
    print(txt_name)
    with open(txt_name, 'r') as file:
        lines = file.readlines()
    results = []
    count_delete = 0
    count_all = 0

    for i, line in enumerate(lines):
        if i == 0:
            count = line.split(' ')[1]
        else:
            line_l = line.split(' ')
            if line_l[0] not in select_l:
                select_l.append(line_l[0])
                results.append(line)
            else:
                count_delete = count_delete + 1
            count_all = count_all + 1

    if txt_name_idx == 0:
        results.insert(0, 'count: ' + str(count_all - count_delete) + '\n')
    else:
        results.insert(0, 'count: ' + str(count_all - count_delete) + ' ' + str(round(count_delete/count_all_past,4)) + '\n')
    count_all_past = len(select_l)
    # print(results)
    # print(txt_name.replace('selected_attributes_', 'selected_attributes_incre_'))
    with open(txt_name.replace('selected_attributes_', 'selected_attributes_incre_'), 'w', encoding='utf-8') as file:
        for item in results:
            file.write(item)
    txt_incre_name_l.append(txt_name.replace('selected_attributes_', 'selected_attributes_incre_'))
    # # exit(0)
    # count_ = count_ + 1
    # if count_ == 2:
    #     exit(0)

for txt_id, txt_file in enumerate(txt_incre_name_l):

    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # 读取去掉第一行的内容
    indices = [int(line.split(' ')[0]) for line in lines[1:]]

    data_path = 'data_all'
    acc_l = []
    for i in range(2):
        x_train = np.load(os.path.join(data_path, str(i).zfill(4) + '_x_train.npy'))
        y_train = np.load(os.path.join(data_path, str(i).zfill(4) + '_y_train.npy'))
        x_test = np.load(os.path.join(data_path, str(i).zfill(4) + '_x_test.npy'))
        y_test = np.load(os.path.join(data_path, str(i).zfill(4) + '_y_test.npy'))
        x_valid = np.load(os.path.join(data_path, str(i).zfill(4) + '_x_valid.npy'))
        y_valid = np.load(os.path.join(data_path, str(i).zfill(4) + '_y_valid.npy'))
        # 从 numpy 数组中抽取对应的属性值
        x_train = x_train[:,indices]
        x_test = x_test[:,indices]
        x_valid = x_valid[:,indices]
        # print('x_train', x_train.shape)
        # print('x_test', x_test.shape)
        # print('x_valid', x_valid.shape)
        # 保存到新的文件中
        save_path = 'data_incre__'+ str(percentage_l[txt_id])
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_x_train.npy'), x_train)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_x_valid.npy'), x_valid)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_x_test.npy'), x_test)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_y_train.npy'), y_train)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_y_valid.npy'), y_valid)
        np.save(os.path.join(save_path, str(i).zfill(4) + '_y_test.npy'), y_test)