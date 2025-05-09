# Eigen Learning-based  Metabolic Information Commonality Analysis in Pan-cancer

This repository hosts the official implementation of the methods discussed in our paper  "Eigen learning-based  metabolic information commonality in Pan-cancer".

## Prepare the enviroment

1. Clone this repository and navigate to Eigen-Metabolism folder

```
git clone https://github.com/Orange066/Eigen-Metabolism.git
cd Eigen-Metabolism
```

2. Install Packages 

* We use Anaconda to create enviroment.

```
conda create -n metabolism python=3.9
conda activate metabolism
```

* Install Pytorch 2.4.1, CUDA 12.4 and CUDNN 

```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

* Install Python Packages: 

```
pip install pandas openpyxl tensorboard tqdm scikit-learn matplotlib
```

## Download the datasets

Please download our preprocessed datasets from the [Zenodo repository](https://doi.org/10.5281/zenodo.14027585) or [Hugging Face]([Orange066/Eigen-metabolism · Datasets at Hugging Face](https://huggingface.co/datasets/Orange066/Eigen-metabolism)) and extract it into the `detection/` folder. The data path should be structured as follows:

```
Eigen-Metabolism/
    datasets/
        enzyme/
            data/
            datasets_ori/
    metabolite/
        M_stage/
            data_all/
            data_select/
        N_stage/
            data_all/
            data_select/
        T_stage/
            data_all/
            data_select/
        tumor_diagnosis/
            data_all/
            data_select/
```

## Obtain and verify eigen-metabolites

### 1. Tumor Diagnosis Task

Run the following codes to get eigen-metabolites for tumor diagnosis task:

```
cd metabolite/tumor_diagnosis
sed -i 's/\r//' bash_run_eigen_learning.sh && bash bash_run_eigen_learning.sh    # eigen-learning training for obtaining eigen-metabolites
python print_result_eigen_learning.py    # see the effectiveness of selected eign-metabolites on the newproject/ folder

sed -i 's/\r//' bash_run_valid_eigen_metabolics.sh && bash bash_run_valid_eigen_metabolics.sh   # verify the effectiveness of each group of eign-metabolites
python print_result_valid_egien.py    # see the effectiveness of each group of eign-metabolites on the newproject/ folder
```

### 2. N stage Task

Run the following codes to get eigen-metabolites for N stage Task:

```
cd metabolite/N_stage
sed -i 's/\r//' bash_run_eigen_learning.sh && bash bash_run_eigen_learning.sh    # eigen-learning training for obtaining eigen-metabolites
python print_result_eigen_learning.py    # see the effectiveness of selected eign-metabolites on the newproject/ folder

sed -i 's/\r//' bash_run_valid_eigen_metabolics.sh && bash bash_run_valid_eigen_metabolics.sh   # verify the effectiveness of each group of eign-metabolites
python print_result_valid_egien.py    # see the effectiveness of each group of eign-metabolites on the newproject/ folder
```

### 3. M stage Task

Run the following codes to get eigen-metabolites for M stage Task:

```
cd metabolite/M_stage
sed -i 's/\r//' bash_run_eigen_learning.sh && bash bash_run_eigen_learning.sh    # eigen-learning training for obtaining eigen-metabolites
python print_result_eigen_learning.py    # see the effectiveness of selected eign-metabolites on the newproject/ folder

sed -i 's/\r//' bash_run_valid_eigen_metabolics.sh && bash bash_run_valid_eigen_metabolics.sh   # verify the effectiveness of each group of eign-metabolites
python print_result_valid_egien.py    # see the effectiveness of each group of eign-metabolites on the newproject/ folder
```

### 4. T Stage Task

Run the following codes to get eigen-metabolites for T Stage Task:

```
cd metabolite/T_stage
sed -i 's/\r//' bash_run_eigen_learning.sh && bash bash_run_eigen_learning.sh    # eigen-learning training for obtaining eigen-metabolites
python print_result_eigen_learning.py    # see the effectiveness of selected eign-metabolites on the newproject/ folder

sed -i 's/\r//' bash_run_valid_eigen_metabolics.sh && bash bash_run_valid_eigen_metabolics.sh   # verify the effectiveness of each group of eign-metabolites
python print_result_valid_egien.py    # see the effectiveness of each group of eign-metabolites on the newproject/ folder
```

You can then find the results in the `metabolite/tumor_diagnosis/newproject/`, `metabolite/N_stage/newproject/`, `metabolite/M_stage/newproject/`, `metabolite/T_stage/newproject/` folders, respectively.

## Obtain and verify eigen-enzyme and pan-eigen-enzyme

### 1. Verify eigen-enzyme

Run the following codes to verify eigen-metabolites for 22 different organs and all organs together:

```
cd enzyme 
sed -i 's/\r//' bash_run_valid_eigen_enzyme.sh && bash bash_run_valid_eigen_enzyme.sh    # verify on 22 different organs
sed -i 's/\r//' bash_run_valid_eigen_enzyme_multitask.sh && bash bash_run_valid_eigen_enzyme_multitask.sh    # verify on all organs togerther
python get_valid_eigen_enzyme_results.py
```

### 2. Obtain pan-eigen-enzyme

Run the following codes to get pan-eigen-metabolites and verify the effectiveness of each group of pan-eigen-metabolites:

```
cd enzyme 
sed -i 's/\r//' bash_run_eigen_learning.sh && bash bash_run_eigen_learning.sh    # eigen-learning training for obtaining pan-eigen-enzyme
sed -i 's/\r//' run_Multitask_valid_pan_eigen_enzyme.sh && bash sed -i 's/\r//' run_Multitask_valid_pan_eigen_enzyme.sh    # verify the effectiveness of each group of pan-eign-enzyme
python get_valid_pan_eigen_enzyme_results.py
```

### 3. Compare to random enzyme

Run the following codes to get random enzyme results:

```
cd enzyme 
sed -i 's/\r//' bash_run_random_enzyme.sh && bash bash_run_random_enzyme.sh # verify the effectiveness of random-enzyme
python get_random_enzyme_results.py
```

You can then find the results in the `enzyme/results/` folder.

------

Please contact **[rcheng22@m.fudan.edu.cn](mailto:rcheng22@m.fudan.edu.cn)** if you have questions.