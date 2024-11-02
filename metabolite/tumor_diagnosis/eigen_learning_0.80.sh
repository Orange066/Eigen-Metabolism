python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0000_x_train.npy --save_name base_select_0000__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.80  --model_weight_path newproject/base_select_0000__0.80/model-9.pth --save_path select_results/base_select_0000__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.80/p_x_train.npy --save_name base_select_0000__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.80/p_x_test.npy --save_name base_select_0000__0.80_p --model_weight_path newproject/base_select_0000__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0000__0.80/n_x_train.npy --save_name base_select_0000__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0000__0.80/n_x_test.npy --save_name base_select_0000__0.80_n --model_weight_path newproject/base_select_0000__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0000_x_train.npy --save_name base_select_0000__0.80  --model_weight_path newproject/base_select_0000__0.80/model-9.pth --save_path select_results/base_select_0000__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0001_x_train.npy --save_name base_select_0001__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.80  --model_weight_path newproject/base_select_0001__0.80/model-9.pth --save_path select_results/base_select_0001__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.80/p_x_train.npy --save_name base_select_0001__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.80/p_x_test.npy --save_name base_select_0001__0.80_p --model_weight_path newproject/base_select_0001__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0001__0.80/n_x_train.npy --save_name base_select_0001__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0001__0.80/n_x_test.npy --save_name base_select_0001__0.80_n --model_weight_path newproject/base_select_0001__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0001_x_train.npy --save_name base_select_0001__0.80  --model_weight_path newproject/base_select_0001__0.80/model-9.pth --save_path select_results/base_select_0001__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0002_x_train.npy --save_name base_select_0002__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.80  --model_weight_path newproject/base_select_0002__0.80/model-9.pth --save_path select_results/base_select_0002__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.80/p_x_train.npy --save_name base_select_0002__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.80/p_x_test.npy --save_name base_select_0002__0.80_p --model_weight_path newproject/base_select_0002__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0002__0.80/n_x_train.npy --save_name base_select_0002__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0002__0.80/n_x_test.npy --save_name base_select_0002__0.80_n --model_weight_path newproject/base_select_0002__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0002_x_train.npy --save_name base_select_0002__0.80  --model_weight_path newproject/base_select_0002__0.80/model-9.pth --save_path select_results/base_select_0002__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0003_x_train.npy --save_name base_select_0003__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0003_x_train.npy --save_name base_select_0003__0.80  --model_weight_path newproject/base_select_0003__0.80/model-9.pth --save_path select_results/base_select_0003__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0003__0.80/p_x_train.npy --save_name base_select_0003__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0003__0.80/p_x_test.npy --save_name base_select_0003__0.80_p --model_weight_path newproject/base_select_0003__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0003__0.80/n_x_train.npy --save_name base_select_0003__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0003__0.80/n_x_test.npy --save_name base_select_0003__0.80_n --model_weight_path newproject/base_select_0003__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0003_x_train.npy --save_name base_select_0003__0.80  --model_weight_path newproject/base_select_0003__0.80/model-9.pth --save_path select_results/base_select_0003__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0004_x_train.npy --save_name base_select_0004__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0004_x_train.npy --save_name base_select_0004__0.80  --model_weight_path newproject/base_select_0004__0.80/model-9.pth --save_path select_results/base_select_0004__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0004__0.80/p_x_train.npy --save_name base_select_0004__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0004__0.80/p_x_test.npy --save_name base_select_0004__0.80_p --model_weight_path newproject/base_select_0004__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0004__0.80/n_x_train.npy --save_name base_select_0004__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0004__0.80/n_x_test.npy --save_name base_select_0004__0.80_n --model_weight_path newproject/base_select_0004__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0004_x_train.npy --save_name base_select_0004__0.80  --model_weight_path newproject/base_select_0004__0.80/model-9.pth --save_path select_results/base_select_0004__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0005_x_train.npy --save_name base_select_0005__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0005_x_train.npy --save_name base_select_0005__0.80  --model_weight_path newproject/base_select_0005__0.80/model-9.pth --save_path select_results/base_select_0005__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0005__0.80/p_x_train.npy --save_name base_select_0005__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0005__0.80/p_x_test.npy --save_name base_select_0005__0.80_p --model_weight_path newproject/base_select_0005__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0005__0.80/n_x_train.npy --save_name base_select_0005__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0005__0.80/n_x_test.npy --save_name base_select_0005__0.80_n --model_weight_path newproject/base_select_0005__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0005_x_train.npy --save_name base_select_0005__0.80  --model_weight_path newproject/base_select_0005__0.80/model-9.pth --save_path select_results/base_select_0005__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0006_x_train.npy --save_name base_select_0006__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0006_x_train.npy --save_name base_select_0006__0.80  --model_weight_path newproject/base_select_0006__0.80/model-9.pth --save_path select_results/base_select_0006__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0006__0.80/p_x_train.npy --save_name base_select_0006__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0006__0.80/p_x_test.npy --save_name base_select_0006__0.80_p --model_weight_path newproject/base_select_0006__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0006__0.80/n_x_train.npy --save_name base_select_0006__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0006__0.80/n_x_test.npy --save_name base_select_0006__0.80_n --model_weight_path newproject/base_select_0006__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0006_x_train.npy --save_name base_select_0006__0.80  --model_weight_path newproject/base_select_0006__0.80/model-9.pth --save_path select_results/base_select_0006__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0007_x_train.npy --save_name base_select_0007__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0007_x_train.npy --save_name base_select_0007__0.80  --model_weight_path newproject/base_select_0007__0.80/model-9.pth --save_path select_results/base_select_0007__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0007__0.80/p_x_train.npy --save_name base_select_0007__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0007__0.80/p_x_test.npy --save_name base_select_0007__0.80_p --model_weight_path newproject/base_select_0007__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0007__0.80/n_x_train.npy --save_name base_select_0007__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0007__0.80/n_x_test.npy --save_name base_select_0007__0.80_n --model_weight_path newproject/base_select_0007__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0007_x_train.npy --save_name base_select_0007__0.80  --model_weight_path newproject/base_select_0007__0.80/model-9.pth --save_path select_results/base_select_0007__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0008_x_train.npy --save_name base_select_0008__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0008_x_train.npy --save_name base_select_0008__0.80  --model_weight_path newproject/base_select_0008__0.80/model-9.pth --save_path select_results/base_select_0008__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0008__0.80/p_x_train.npy --save_name base_select_0008__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0008__0.80/p_x_test.npy --save_name base_select_0008__0.80_p --model_weight_path newproject/base_select_0008__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0008__0.80/n_x_train.npy --save_name base_select_0008__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0008__0.80/n_x_test.npy --save_name base_select_0008__0.80_n --model_weight_path newproject/base_select_0008__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0008_x_train.npy --save_name base_select_0008__0.80  --model_weight_path newproject/base_select_0008__0.80/model-9.pth --save_path select_results/base_select_0008__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0009_x_train.npy --save_name base_select_0009__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0009_x_train.npy --save_name base_select_0009__0.80  --model_weight_path newproject/base_select_0009__0.80/model-9.pth --save_path select_results/base_select_0009__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0009__0.80/p_x_train.npy --save_name base_select_0009__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0009__0.80/p_x_test.npy --save_name base_select_0009__0.80_p --model_weight_path newproject/base_select_0009__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0009__0.80/n_x_train.npy --save_name base_select_0009__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0009__0.80/n_x_test.npy --save_name base_select_0009__0.80_n --model_weight_path newproject/base_select_0009__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0009_x_train.npy --save_name base_select_0009__0.80  --model_weight_path newproject/base_select_0009__0.80/model-9.pth --save_path select_results/base_select_0009__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0010_x_train.npy --save_name base_select_0010__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0010_x_train.npy --save_name base_select_0010__0.80  --model_weight_path newproject/base_select_0010__0.80/model-9.pth --save_path select_results/base_select_0010__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0010__0.80/p_x_train.npy --save_name base_select_0010__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0010__0.80/p_x_test.npy --save_name base_select_0010__0.80_p --model_weight_path newproject/base_select_0010__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0010__0.80/n_x_train.npy --save_name base_select_0010__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0010__0.80/n_x_test.npy --save_name base_select_0010__0.80_n --model_weight_path newproject/base_select_0010__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0010_x_train.npy --save_name base_select_0010__0.80  --model_weight_path newproject/base_select_0010__0.80/model-9.pth --save_path select_results/base_select_0010__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0011_x_train.npy --save_name base_select_0011__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0011_x_train.npy --save_name base_select_0011__0.80  --model_weight_path newproject/base_select_0011__0.80/model-9.pth --save_path select_results/base_select_0011__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0011__0.80/p_x_train.npy --save_name base_select_0011__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0011__0.80/p_x_test.npy --save_name base_select_0011__0.80_p --model_weight_path newproject/base_select_0011__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0011__0.80/n_x_train.npy --save_name base_select_0011__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0011__0.80/n_x_test.npy --save_name base_select_0011__0.80_n --model_weight_path newproject/base_select_0011__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0011_x_train.npy --save_name base_select_0011__0.80  --model_weight_path newproject/base_select_0011__0.80/model-9.pth --save_path select_results/base_select_0011__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0012_x_train.npy --save_name base_select_0012__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0012_x_train.npy --save_name base_select_0012__0.80  --model_weight_path newproject/base_select_0012__0.80/model-9.pth --save_path select_results/base_select_0012__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0012__0.80/p_x_train.npy --save_name base_select_0012__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0012__0.80/p_x_test.npy --save_name base_select_0012__0.80_p --model_weight_path newproject/base_select_0012__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0012__0.80/n_x_train.npy --save_name base_select_0012__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0012__0.80/n_x_test.npy --save_name base_select_0012__0.80_n --model_weight_path newproject/base_select_0012__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0012_x_train.npy --save_name base_select_0012__0.80  --model_weight_path newproject/base_select_0012__0.80/model-9.pth --save_path select_results/base_select_0012__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0013_x_train.npy --save_name base_select_0013__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0013_x_train.npy --save_name base_select_0013__0.80  --model_weight_path newproject/base_select_0013__0.80/model-9.pth --save_path select_results/base_select_0013__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0013__0.80/p_x_train.npy --save_name base_select_0013__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0013__0.80/p_x_test.npy --save_name base_select_0013__0.80_p --model_weight_path newproject/base_select_0013__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0013__0.80/n_x_train.npy --save_name base_select_0013__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0013__0.80/n_x_test.npy --save_name base_select_0013__0.80_n --model_weight_path newproject/base_select_0013__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0013_x_train.npy --save_name base_select_0013__0.80  --model_weight_path newproject/base_select_0013__0.80/model-9.pth --save_path select_results/base_select_0013__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0014_x_train.npy --save_name base_select_0014__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0014_x_train.npy --save_name base_select_0014__0.80  --model_weight_path newproject/base_select_0014__0.80/model-9.pth --save_path select_results/base_select_0014__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0014__0.80/p_x_train.npy --save_name base_select_0014__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0014__0.80/p_x_test.npy --save_name base_select_0014__0.80_p --model_weight_path newproject/base_select_0014__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0014__0.80/n_x_train.npy --save_name base_select_0014__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0014__0.80/n_x_test.npy --save_name base_select_0014__0.80_n --model_weight_path newproject/base_select_0014__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0014_x_train.npy --save_name base_select_0014__0.80  --model_weight_path newproject/base_select_0014__0.80/model-9.pth --save_path select_results/base_select_0014__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0015_x_train.npy --save_name base_select_0015__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0015_x_train.npy --save_name base_select_0015__0.80  --model_weight_path newproject/base_select_0015__0.80/model-9.pth --save_path select_results/base_select_0015__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0015__0.80/p_x_train.npy --save_name base_select_0015__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0015__0.80/p_x_test.npy --save_name base_select_0015__0.80_p --model_weight_path newproject/base_select_0015__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0015__0.80/n_x_train.npy --save_name base_select_0015__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0015__0.80/n_x_test.npy --save_name base_select_0015__0.80_n --model_weight_path newproject/base_select_0015__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0015_x_train.npy --save_name base_select_0015__0.80  --model_weight_path newproject/base_select_0015__0.80/model-9.pth --save_path select_results/base_select_0015__0.80  --device cuda:0

python newtrain.py --ismask --isresult --sparse_loss_w 5.0 --epochs 10 --tgt_sparse 0.80 --x_train_path data_select/0016_x_train.npy --save_name base_select_0016__0.80  --device cuda:0

python newpredict.py --ismask --x_test_path data_select/0016_x_train.npy --save_name base_select_0016__0.80  --model_weight_path newproject/base_select_0016__0.80/model-9.pth --save_path select_results/base_select_0016__0.80  --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0016__0.80/p_x_train.npy --save_name base_select_0016__0.80_p --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0016__0.80/p_x_test.npy --save_name base_select_0016__0.80_p --model_weight_path newproject/base_select_0016__0.80_p/model-val-auc-best.pth --device cuda:0

python newtrain.py --epochs 10 --x_train_path select_results/base_select_0016__0.80/n_x_train.npy --save_name base_select_0016__0.80_n --device cuda:0

python newpredict.py --isresult --x_test_path select_results/base_select_0016__0.80/n_x_test.npy --save_name base_select_0016__0.80_n --model_weight_path newproject/base_select_0016__0.80_n/model-val-auc-best.pth --device cuda:0

python newpredict.py --ismask --important_test --x_test_path data_select/0016_x_train.npy --save_name base_select_0016__0.80  --model_weight_path newproject/base_select_0016__0.80/model-9.pth --save_path select_results/base_select_0016__0.80  --device cuda:0

