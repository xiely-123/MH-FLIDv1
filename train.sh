python main_RHM-FedTS_OCT.py  --task=OCT  --epochs=200 --local_ep=1 --local_TS_ep=1 --gpu=0 --num_layers_keep=13 --lr=0.0001  --lr_s=0.00001 --num_users=8 --local_bs=8 --iid=0


python main_RHM-FedTS_1.py  --task=MH-breast-HR  --epochs=200 --local_ep=4 --local_TS_ep=1 --gpu=2 --num_layers_keep=13 --lr=0.0001 --lr_s=0.00001 --num_users=8 --local_bs=8 --iid=0

python main_RHM-FedTS_Timeseries.py  --task=Time-series  --epochs=200 --local_ep=4 --local_TS_ep=1 --gpu=2 --num_layers_keep=13 --lr=0.001  --lr_s=0.00001 --num_users=3 --local_bs=32 --iid=0


python main_RHM-FedTS_segementation.py  --task=Segment  --epochs=200 --local_ep=4 --local_TS_ep=1 --gpu=0 --num_layers_keep=21 --lr=0.0001 --lr_s=0.00001 --num_users=4 --local_bs=8 --iid=0
