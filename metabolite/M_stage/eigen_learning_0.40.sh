python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.40 --x_train_path data_select/0000_x_train.npy --save_name base_select_0000__0.40  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.40  --model_weight_path newproject/base_select_0000__0.40/model-9.pth --save_path select_results/base_select_0000__0.40  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.40/p_x_train.npy --save_name base_select_0000__0.40_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.40/p_x_test.npy --save_name base_select_0000__0.40_p --model_weight_path newproject/base_select_0000__0.40_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.40/n_x_train.npy --save_name base_select_0000__0.40_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.40/n_x_test.npy --save_name base_select_0000__0.40_n --model_weight_path newproject/base_select_0000__0.40_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.40  --model_weight_path newproject/base_select_0000__0.40/model-9.pth --save_path select_results/base_select_0000__0.40  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.40 --x_train_path data_select/0001_x_train.npy --save_name base_select_0001__0.40  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.40  --model_weight_path newproject/base_select_0001__0.40/model-9.pth --save_path select_results/base_select_0001__0.40  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.40/p_x_train.npy --save_name base_select_0001__0.40_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.40/p_x_test.npy --save_name base_select_0001__0.40_p --model_weight_path newproject/base_select_0001__0.40_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.40/n_x_train.npy --save_name base_select_0001__0.40_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.40/n_x_test.npy --save_name base_select_0001__0.40_n --model_weight_path newproject/base_select_0001__0.40_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.40  --model_weight_path newproject/base_select_0001__0.40/model-9.pth --save_path select_results/base_select_0001__0.40  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.40 --x_train_path data_select/0002_x_train.npy --save_name base_select_0002__0.40  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.40  --model_weight_path newproject/base_select_0002__0.40/model-9.pth --save_path select_results/base_select_0002__0.40  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.40/p_x_train.npy --save_name base_select_0002__0.40_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.40/p_x_test.npy --save_name base_select_0002__0.40_p --model_weight_path newproject/base_select_0002__0.40_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.40/n_x_train.npy --save_name base_select_0002__0.40_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.40/n_x_test.npy --save_name base_select_0002__0.40_n --model_weight_path newproject/base_select_0002__0.40_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.40  --model_weight_path newproject/base_select_0002__0.40/model-9.pth --save_path select_results/base_select_0002__0.40  --device cuda:0

