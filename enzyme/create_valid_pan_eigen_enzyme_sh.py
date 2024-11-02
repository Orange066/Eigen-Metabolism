import os

organ_names = [
    'Multitask',

]

task_l = ['tumor']
cuda_l = [
    '0',
]
subpaths = ['tumor_mask_0.0042', 'tumor_mask_0.005', 'tumor_mask_0.02', 'tumor_mask_0.03', 'tumor_mask_0.05']
data_base = 'data'

for organ_i, organ in enumerate(organ_names):
    f = open('run_' + organ + '_valid_pan_eigen_enzyme.sh', 'w')
    for task_i, task in enumerate(task_l):
        for i in range(1, len(subpaths)):
            w_str = 'CUDA_VISIBLE_DEVICES=' + cuda_l[
                organ_i] + ' python experiments_incre.py ' + '--task ' + task + ' --run_type incre_all ' + ' --organ ' + organ + ' --large_model --incre_all_idx ' +  str(i)
            f.write(w_str + '\n\n')

        for i in range(len(subpaths)):
            for j in range(i + 1, len(subpaths)):

                w_str = 'CUDA_VISIBLE_DEVICES=' + cuda_l[
                    organ_i] + ' python experiments_incre.py ' + '--task ' + task + ' --run_type incre_single ' + ' --organ ' + organ + ' --large_model --incre_single_idx ' + str(
                    i).zfill(4) + '_' + str(j).zfill(4)
                f.write(w_str + '\n\n')


    f.close()