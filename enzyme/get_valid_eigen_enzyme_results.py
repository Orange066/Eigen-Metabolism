import os.path

import pandas as pd
import openpyxl

organ_names = [
    'Adrena-Gland',
    'Bile-Duct',
    'Bladder',
    'Brain',
    'Breast',

    'Cervix',
    'Colon',
    'Esophagus',
    'Gliomas',

    'Liver',
    'Lung',
    'LungSquamous',
    'Neck',
    'Ovary',

    'Pancreas',
    'Prostate',
    'Rectum',
    'Renal-Papillae',
    'Renal-Transparency',

    'Skin',
    'Stomach',
    'Testis',
    'Thyroid',
    'Uterus',

    'Multitask'
]


task_l = ['tumor', 'transfer', 'further', 'time']
base_result = 'results'
for organ in organ_names:
    # if organ != 'Pancreas':
    #     continue
    for task in task_l:
        print(task)
        # 读取Excel文件
        auc_sum = 0
        acc_sum = 0
        f1_sum = 0
        recall_sum = 0
        prec_sum = 0
        results_l = []
        cells = ['', 'AUC', 'F1', 'ACC', 'PREC', 'REC']
        results_l.append(cells)

        file_path = os.path.join(base_result, organ, task + '_select', 'results.xlsx')
        print()
        if os.path.exists(file_path) == False:
            print('task', task, 'continue')
            continue
        df = pd.read_excel(file_path)
        last_row = df.iloc[-1]
        last_row[0] = 'Select'
        results_l.append(last_row)

        file_path = os.path.join(base_result, organ, task + '_all', 'results.xlsx')
        df = pd.read_excel(file_path)
        last_row = df.iloc[-1]
        last_row[0] = 'All'
        results_l.append(last_row)

        # for i in range(5):
        #     file_path = os.path.join(base_result, organ, task+'_random_'+str(i).zfill(4), 'results.xlsx')
        #     df = pd.read_excel(file_path)
        #     last_row = df.iloc[-1]
        #     auc_sum += last_row[1]
        #     acc_sum += last_row[2]
        #     f1_sum += last_row[3]
        #     prec_sum += last_row[4]
        #     recall_sum += last_row[5]
        #
        # auc_sum = "{:.4f}".format(auc_sum/5)
        # acc_sum = "{:.4f}".format(acc_sum/5)
        # f1_sum = "{:.4f}".format(f1_sum/5)
        # recall_sum = "{:.4f}".format(recall_sum/5)
        # prec_sum = "{:.4f}".format(prec_sum/5)
        # cells = ['Random', auc_sum, acc_sum, f1_sum, prec_sum, recall_sum]
        # results_l.append(cells)

        # file_path = os.path.join(base_result, organ, task + '_mask', 'results.xlsx')
        # if os.path.exists(file_path):
        #     df = pd.read_excel(file_path)
        #     last_row = df.iloc[-1]
        #     last_row[0] = 'Mask'
        #     results_l.append(last_row)


        # name_l = ['80%', '60%', '40%', '20%']
        # for i in range(4):
        #     file_path = os.path.join(base_result, organ, task+'_prop_'+str(i).zfill(4), 'results.xlsx')
        #     df = pd.read_excel(file_path)
        #     last_row = df.iloc[-1]
        #     last_row[0] = name_l[i]
        #     results_l.append(last_row)

        wb = openpyxl.Workbook()
        ws = wb.active

        for row_index, data_list in enumerate(results_l, start=1):
            for col_index, value in enumerate(data_list, start=1):
                if isinstance(value, (float, int)):  # 仅格式化数值类型
                    value = "{:.4f}".format(value)
                ws.cell(row=row_index, column=col_index, value=value)
        wb.save(os.path.join(base_result, organ, ''+ task + '.xlsx'))
        print(os.path.join(base_result, organ, ''+ task + '.xlsx'))