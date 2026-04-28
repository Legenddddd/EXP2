<div align="center">    
 
# Progressively exploring and exploiting inference data to break fine-grained classification barriers
Li-Jun Zhao, Si-Yuan Zhang, Zhen-Duo Chen, Xin Luo, and Xin-Shun Xu

</div>

```bibtex
@article{Zhao_2026,
   title={突破细粒度分类障碍的渐进式推理阶段数据探索与利用方法},
   ISSN={1674-7267},
   url={http://dx.doi.org/10.1360/SSI-2025-0502},
   DOI={10.1360/ssi-2025-0502},
   journal={SCIENTIA SINICA Informationis},
   publisher={Science China Press., Co. Ltd.},
   author={Zhao, Lijun and Siyuan, Zhang and Chen, Zhenduo and Luo, Xin and Xu, Xinshun},
   year={2026},
   month=Feb }
```

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
