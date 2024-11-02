python newpredict.py --ismask --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.20  --model_weight_path newproject/base_select_0000__0.20/model-9.pth --save_path select_results/base_select_0000__0.20  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.20/p_x_train.npy --save_name base_select_0000__0.20_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.20/p_x_test.npy --save_name base_select_0000__0.20_p --model_weight_path newproject/base_select_0000__0.20_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.20/n_x_train.npy --save_name base_select_0000__0.20_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.20/n_x_test.npy --save_name base_select_0000__0.20_n --model_weight_path newproject/base_select_0000__0.20_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.20  --model_weight_path newproject/base_select_0000__0.20/model-9.pth --save_path select_results/base_select_0000__0.20  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.20  --model_weight_path newproject/base_select_0001__0.20/model-9.pth --save_path select_results/base_select_0001__0.20  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.20/p_x_train.npy --save_name base_select_0001__0.20_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.20/p_x_test.npy --save_name base_select_0001__0.20_p --model_weight_path newproject/base_select_0001__0.20_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.20/n_x_train.npy --save_name base_select_0001__0.20_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.20/n_x_test.npy --save_name base_select_0001__0.20_n --model_weight_path newproject/base_select_0001__0.20_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.20  --model_weight_path newproject/base_select_0001__0.20/model-9.pth --save_path select_results/base_select_0001__0.20  --device cuda:0

