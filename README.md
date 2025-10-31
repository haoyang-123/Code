# Code
Code for Plos One
## Environment

- python == 3.8.10
- pytorch == 1.10.0
- networkx == 2.5
- numpy == 1.20.2

GPU: NVIDIA RTX 3090 GPU 

### Train and Test Our proposed model

```python
python train_others.py --model GCN --hidden 64 --dataset dataset --labelrate labelrate --stage 1

- **dataset:**  including [Cora, Citeseer, Pubmed, CoraFull], required.
- **labelrate:** including [20, 40, 60], required.
```
e.g.,

```
python train_others.py --model GCN --hidden 64 --dataset Cora --labelrate 20 --stage 1
python train_others.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --stage 1 --dropout 0.5 --lr 0.005
```

### Temperature scaling & Matring Scaling

```python
python train_others.py --model GCN --scaling_method method --hidden 64 --dataset dataset --labelrate labelrate --stage 1 --lr_for_cal 0.01 --max_iter 50
python train_others.py --model GAT --scaling_method method --hidden 8 --dataset dataset --labelrate labelrate --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --max_iter 50
```

- **method:** including [TS, MS], required.
- **dataset:**  including [Cora, Citeseer, Pubmed, CoraFull], required.
- **labelrate:** including [20, 40, 60], required.

e.g.,

```
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Cora --labelrate 20 --stage 1 --lr_for_cal 0.01 --max_iter 50
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --max_iter 50
