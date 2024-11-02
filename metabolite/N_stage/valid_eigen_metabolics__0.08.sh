python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0000_x_train.npy --save_name base_incre_0000___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0000_x_test.npy --save_name base_incre_0000___0.08  --model_weight_path newproject/base_incre_0000___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0000___0.08  --device cuda:0

python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0001_x_train.npy --save_name base_incre_0001___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0001_x_test.npy --save_name base_incre_0001___0.08  --model_weight_path newproject/base_incre_0001___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0001___0.08  --device cuda:0

python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0002_x_train.npy --save_name base_incre_0002___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0002_x_test.npy --save_name base_incre_0002___0.08  --model_weight_path newproject/base_incre_0002___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0002___0.08  --device cuda:0

python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0003_x_train.npy --save_name base_incre_0003___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0003_x_test.npy --save_name base_incre_0003___0.08  --model_weight_path newproject/base_incre_0003___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0003___0.08  --device cuda:0

python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0004_x_train.npy --save_name base_incre_0004___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0004_x_test.npy --save_name base_incre_0004___0.08  --model_weight_path newproject/base_incre_0004___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0004___0.08  --device cuda:0

python newtrain.py --sparse_loss_w 5.0 --epochs 10 --x_train_path data_incre__0.08/0005_x_train.npy --save_name base_incre_0005___0.08  --device cuda:0

python newpredict.py --isresult --x_test_path data_incre__0.08/0005_x_test.npy --save_name base_incre_0005___0.08  --model_weight_path newproject/base_incre_0005___0.08/model-val-auc-best.pth --save_path select_results/base_incre_0005___0.08  --device cuda:0

