import os


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
]

task_l = ['tumor', 'transfer', 'further', 'time']
cuda_l = [
    '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0',
]


data_base = 'data'
save_shs = []
for organ_i, organ in enumerate(organ_names):
    save_shs.append('run_' + organ + '.sh')
    f = open('run_' + organ + '.sh', 'w')
    for task_i, task in enumerate(task_l):

        # if task == 'tumor':
        #     file1_path = os.path.join(data_base, 'Tumor', 'GTEx-TCGA-' + organ + '-normal-select.npz')
        #     if os.path.exists(file1_path) == False:
        #         continue
        # else:
        file1_path = os.path.join(data_base, organ, task + '_positive_samples_tpms_prop_0003.npz')
        if os.path.exists(file1_path) == False:
            continue

        w_str = 'CUDA_VISIBLE_DEVICES='+ cuda_l[organ_i] +' python experiments.py ' + '--task ' + task + ' --run_type select ' + ' --organ ' + organ
        f.write(w_str + '\n\n')
        w_str = 'CUDA_VISIBLE_DEVICES='+ cuda_l[organ_i] +' python experiments.py ' + '--task ' + task + ' --run_type all ' + ' --organ ' + organ
        f.write(w_str + '\n\n')
        # for i in range(5):
        #     w_str = 'CUDA_VISIBLE_DEVICES='+ cuda_l[organ_i] +' python experiments.py ' + '--task ' + task + ' --run_type random --random_idx '  + str(i)  + ' --organ ' + organ
        #     f.write(w_str + '\n\n')
        # for i in range(4):
        #     w_str = 'CUDA_VISIBLE_DEVICES='+ cuda_l[organ_i] +' python experiments.py ' + '--task ' + task + ' --run_type prop --prop_idx '  + str(i)  + ' --organ ' + organ
        #     f.write(w_str + '\n\n')
        # w_str = 'CUDA_VISIBLE_DEVICES='+ cuda_l[organ_i] +' python experiments.py ' + '--task ' + task + ' --run_type mask '  + ' --organ ' + organ
        # f.write(w_str + '\n\n')

    f.close()

with open('bash_run_valid_eigen_enzyme.sh', 'w') as file:
    for sh  in save_shs:
        w_str = "sed -i 's/\\r//' " + sh + ' && bash '+ sh
        file.write(w_str + '\n\n')