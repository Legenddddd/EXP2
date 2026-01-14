We provide the code for our EXP2 strategy. 

## Requirements
- [PyTorch >= version 1.1](https://pytorch.org)
- tqdm

## Datasets
We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  For newly introduced fine-grained datasets, we also provide the corresponding index_list.

Please download the dataset before you run the code. For CIFAR100, the dataset will be download automatically.  



## Scripts
Running the shell script ```run.sh``` will train and evaluate the model with hyperparameters matching our paper.

    
## Acknowledgment
Our project references the codes in the following repos.

- [CEC](https://github.com/xyutao/fscil)
