python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0000_x_train.npy --save_name base_select_0000__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.01  --model_weight_path newproject/base_select_0000__0.01/model-9.pth --save_path select_results/base_select_0000__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.01/p_x_train.npy --save_name base_select_0000__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.01/p_x_test.npy --save_name base_select_0000__0.01_p --model_weight_path newproject/base_select_0000__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.01/n_x_train.npy --save_name base_select_0000__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.01/n_x_test.npy --save_name base_select_0000__0.01_n --model_weight_path newproject/base_select_0000__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.01  --model_weight_path newproject/base_select_0000__0.01/model-9.pth --save_path select_results/base_select_0000__0.01  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0001_x_train.npy --save_name base_select_0001__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.01  --model_weight_path newproject/base_select_0001__0.01/model-9.pth --save_path select_results/base_select_0001__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.01/p_x_train.npy --save_name base_select_0001__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.01/p_x_test.npy --save_name base_select_0001__0.01_p --model_weight_path newproject/base_select_0001__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.01/n_x_train.npy --save_name base_select_0001__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.01/n_x_test.npy --save_name base_select_0001__0.01_n --model_weight_path newproject/base_select_0001__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.01  --model_weight_path newproject/base_select_0001__0.01/model-9.pth --save_path select_results/base_select_0001__0.01  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0002_x_train.npy --save_name base_select_0002__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.01  --model_weight_path newproject/base_select_0002__0.01/model-9.pth --save_path select_results/base_select_0002__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.01/p_x_train.npy --save_name base_select_0002__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.01/p_x_test.npy --save_name base_select_0002__0.01_p --model_weight_path newproject/base_select_0002__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.01/n_x_train.npy --save_name base_select_0002__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.01/n_x_test.npy --save_name base_select_0002__0.01_n --model_weight_path newproject/base_select_0002__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.01  --model_weight_path newproject/base_select_0002__0.01/model-9.pth --save_path select_results/base_select_0002__0.01  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0003_x_train.npy --save_name base_select_0003__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0003_x_train.npy --save_name base_select_0003__0.01  --model_weight_path newproject/base_select_0003__0.01/model-9.pth --save_path select_results/base_select_0003__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0003__0.01/p_x_train.npy --save_name base_select_0003__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0003__0.01/p_x_test.npy --save_name base_select_0003__0.01_p --model_weight_path newproject/base_select_0003__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0003__0.01/n_x_train.npy --save_name base_select_0003__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0003__0.01/n_x_test.npy --save_name base_select_0003__0.01_n --model_weight_path newproject/base_select_0003__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0003_x_train.npy --save_name base_select_0003__0.01  --model_weight_path newproject/base_select_0003__0.01/model-9.pth --save_path select_results/base_select_0003__0.01  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0004_x_train.npy --save_name base_select_0004__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0004_x_train.npy --save_name base_select_0004__0.01  --model_weight_path newproject/base_select_0004__0.01/model-9.pth --save_path select_results/base_select_0004__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0004__0.01/p_x_train.npy --save_name base_select_0004__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0004__0.01/p_x_test.npy --save_name base_select_0004__0.01_p --model_weight_path newproject/base_select_0004__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0004__0.01/n_x_train.npy --save_name base_select_0004__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0004__0.01/n_x_test.npy --save_name base_select_0004__0.01_n --model_weight_path newproject/base_select_0004__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0004_x_train.npy --save_name base_select_0004__0.01  --model_weight_path newproject/base_select_0004__0.01/model-9.pth --save_path select_results/base_select_0004__0.01  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.01 --x_train_path data_select/0005_x_train.npy --save_name base_select_0005__0.01  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0005_x_train.npy --save_name base_select_0005__0.01  --model_weight_path newproject/base_select_0005__0.01/model-9.pth --save_path select_results/base_select_0005__0.01  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0005__0.01/p_x_train.npy --save_name base_select_0005__0.01_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0005__0.01/p_x_test.npy --save_name base_select_0005__0.01_p --model_weight_path newproject/base_select_0005__0.01_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0005__0.01/n_x_train.npy --save_name base_select_0005__0.01_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0005__0.01/n_x_test.npy --save_name base_select_0005__0.01_n --model_weight_path newproject/base_select_0005__0.01_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0005_x_train.npy --save_name base_select_0005__0.01  --model_weight_path newproject/base_select_0005__0.01/model-9.pth --save_path select_results/base_select_0005__0.01  --device cuda:0

