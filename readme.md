# Fusion of Global and Local Knowledge for  Personalized Federated Learning 
This is the repo for the paper "Fusion of Global and Local Knowledge for Personalized Federated Learning".

## Algorithm overview
The overall procedure can be summarized into four main steps. i) Training of global knowledge representation with auxiliary variables,  ii) local fusion with sparse personalized component, 
iii) auxiliary variables update,  and iv) server update with proximal step. 
The following figure illustrates the overall process. 
<div align=center><img width="700" height="450" src="https://github.com/huangtiansheng/fedslr/blob/main/fig/fedslr.png"/></div>


##Baselines
* FedAvg (http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
* SCAFFOLD (http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf)
* APFL (https://arxiv.org/pdf/2003.13461.pdf)
* Ditto (http://proceedings.mlr.press/v139/li21h/li21h.pdf)

## Code structure
1. Methods in `utils_methods.py` captures the server logic in FL. 
2. Methods in `utils_general.py` includes the client logic, which normally would be called by a method in `utils_methods.py`.
3. Packages are imported  in `utils_lib.py`.  
4. Models are created in `utils_models.py`.
5. Data is splitted and prepared in `utils_dataset.py`.
6. Main entrance is in `train.py`.


## Run Instruction
We have prepared a script for an easy run. To test the code, simply input the following command...
1. `cd example` 
2. `nohup .runner.sh &` 

You can also run via entering separate command, e.g.,  
`nohup python train.py  --non_iid --method FedSLR --dataset CIFAR100 --gpu 0  &`
## Log Format
* Testing would be conducted each 5 communication rounds, with the following format:  
`**** Cur  w model   10, Test Accuracy: 0.0369, Loss: 4.4591  `  
`**** Cur low rank + sparse model  10, Test Accuracy: 0.1482, Loss: 3.4453`  
  The acc in the above line is the accuracy of low-rank model, and acc in the bottom line is the acc of personalized models.
  
## Citation

If you find this repo useful, please cite our paper:

```
@article{huang2022fusion,
  title={Fusion of Global and Local Knowledge for Personalized Federated Learning},
  author={Huang, Tiansheng and Shen, Li and Sun, Yan and Lin, Weiwei and Tao, Dacheng},
}
```

