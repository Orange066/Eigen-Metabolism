import os

save_name_l = ['eigen_learning']
c1_save = 'select_'
tgt_sparse_l = ['0.01', '0.05', '0.10', '0.20', '0.40', '0.60', '0.80']
command_l = ['--sparse_loss_w 5.0 --epochs 10 ']
device_l = ['cuda:0', 'cuda:0','cuda:0', 'cuda:0','cuda:0', 'cuda:0','cuda:0']
save_path_l = ['']

base_command_1 = 'python newtrain.py --ismask --isresult '
# --important_test
base_command_2 = 'python newpredict.py --ismask '
base_command_3 = 'python newtrain.py --epochs 10 '
base_command_4= 'python newpredict.py --isresult '
base_command_5 = 'python newtrain.py --epochs 10 '
base_command_6= 'python newpredict.py --isresult '
base_command_7 = 'python newpredict.py --ismask --important_test '

save_shs = []
for save_name_id, save_name in enumerate(save_name_l):
    for tgt_sparse_id, tgt_sparse in enumerate(tgt_sparse_l):
            save_shs.append(save_name + '_' + tgt_sparse + '.sh')
            with open(save_name + '_' + tgt_sparse + '.sh', 'w') as file:
                for i in range(2):
                        w_str = base_command_1 + command_l[
                        save_name_id] + '--tgt_sparse ' + tgt_sparse + ' ' + '--x_train_path data_select/' + str(i).zfill(
                        4) + '_x_train.npy --save_name base_'+c1_save + str(i).zfill(4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] +' ' + ' --device ' + device_l[save_name_id*2+tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_2 + '--x_test_path data_select/' + str(i).zfill(
                            4) + '_x_train.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] +' ' + ' --model_weight_path newproject/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/model-9.pth --save_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] +' ' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_3 + '--x_train_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/p_x_train.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_p' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_4 + '--x_test_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/p_x_test.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_p --model_weight_path newproject/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_p/model-val-auc-best.pth' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_5 + '--x_train_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/n_x_train.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_n' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_6 + '--x_test_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/n_x_test.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_n --model_weight_path newproject/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '_n/model-val-auc-best.pth' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_7 + '--x_test_path data_select/' + str(i).zfill(
                            4) + '_x_train.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] +' ' + ' --model_weight_path newproject/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] + '/model-9.pth --save_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + tgt_sparse_l[tgt_sparse_id] +' ' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')

with open('bash_run_eigen_learning.sh', 'w') as file:
    for sh  in save_shs:
        w_str = "sed -i 's/\\r//' " + sh + ' && bash '+ sh
        file.write(w_str + '\n\n')
