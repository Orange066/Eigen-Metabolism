import os

save_name_l = ['valid_eigen_metabolics']
c1_save = 'incre_'
# tgt_sparse_l = ['1.00']
command_l = ['--sparse_loss_w 5.0 --epochs 10 ']
device_l = ['cuda:0','cuda:0','cuda:0','cuda:0','cuda:0','cuda:0','cuda:0']
save_path_l = ['']
percentage_l=['_0.002', '_0.01', '_0.021', '_0.023', '_0.04', '_0.06', '_0.1']
base_command_1 = 'python newtrain.py '
# --important_test
base_command_2 = 'python newpredict.py --isresult '

save_shs = []
for save_name_id, save_name in enumerate(save_name_l):
    for tgt_sparse_id, tgt_sparse in enumerate(percentage_l):
            save_shs.append(save_name + '_' + tgt_sparse + '.sh')
            with open(save_name + '_' + tgt_sparse + '.sh', 'w') as file:
                for i in range(2):
                        w_str = base_command_1 + command_l[
                        save_name_id]  + '--x_train_path data_incre_'+percentage_l[tgt_sparse_id] + '/' + str(i).zfill(
                        4) + '_x_train.npy --save_name base_'+c1_save + str(i).zfill(4) + '_' + save_path_l[save_name_id] +'_' + percentage_l[tgt_sparse_id] +' ' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')
                        w_str = base_command_2 + '--x_test_path data_incre_'+percentage_l[tgt_sparse_id] + '/' + str(i).zfill(
                            4) + '_x_test.npy --save_name base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + percentage_l[tgt_sparse_id] +' ' + ' --model_weight_path newproject/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + percentage_l[tgt_sparse_id] + '/model-val-auc-best.pth --save_path select_results/base_'+c1_save + str(i).zfill(
                            4) + '_' + save_path_l[save_name_id] +'_' + percentage_l[tgt_sparse_id] +' ' + ' --device ' + device_l[tgt_sparse_id]
                        file.write(w_str + '\n\n')

with open('bash_run_valid_eigen_metabolics.sh', 'w') as file:
    for sh  in save_shs:
        w_str = "sed -i 's/\\r//' " + sh + ' && bash '+ sh
        file.write(w_str + '\n\n')