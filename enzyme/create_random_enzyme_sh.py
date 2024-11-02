import os

organ_names = [
    'Multitask',

]
# 4, 10, 19, 28
task_l = ['tumor']
select_num_l = [2, 5, 9, 16, 26]
cuda_l = [
    '0','0','0','0','0'
]

data_base = 'data'
save_shs = []
for organ_i, organ in enumerate(organ_names):
    for select_num_idx, select_num in enumerate(select_num_l):
        save_shs.append('run_' + organ + '_random_select_in_' + str(select_num_idx) + '.sh')
        f = open('run_' + organ + '_random_select_in_' + str(select_num_idx) + '.sh', 'w')
        for task_i, task in enumerate(task_l):
            for i in range(10):
                w_str = 'CUDA_VISIBLE_DEVICES=' + cuda_l[
                    select_num_idx] + ' python experiments_select_random.py ' + '--task ' + task + ' --run_type random_select_in --random_idx ' + str(
                    i) + ' --organ ' + organ + ' --large_model' + ' --select_num ' + str(select_num)
                f.write(w_str + '\n\n')


        f.close()

task_l = ['tumor']
select_num_l = [2, 5, 9, 16, 26]
cuda_l = [
    '0','0','0','0','0'
]

data_base = 'data'

for organ_i, organ in enumerate(organ_names):
    for select_num_idx, select_num in enumerate(select_num_l):
        save_shs.append('run_' + organ + '_random_select_' + str(select_num_idx) + '.sh')
        f = open('run_' + organ + '_random_select_' + str(select_num_idx) + '.sh', 'w')
        for task_i, task in enumerate(task_l):
            for i in range(100):
                w_str = 'CUDA_VISIBLE_DEVICES=' + cuda_l[
                    select_num_idx] + ' python experiments_select_random.py ' + '--task ' + task + ' --run_type random_select --random_idx ' + str(
                    i) + ' --organ ' + organ + ' --large_model' + ' --select_num ' + str(select_num)
                f.write(w_str + '\n\n')


        f.close()

with open('bash_run_random_enzyme.sh', 'w') as file:
    for sh  in save_shs:
        w_str = "sed -i 's/\\r//' " + sh + ' && bash '+ sh
        file.write(w_str + '\n\n')