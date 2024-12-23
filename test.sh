python train_my.py rpknet --way autocast -b 2 --pyramid_ranges 4 2 --iters 12 --corr_mode allpairs --epochs 110 --not_cache_pkconv_weights --enc_norm_type batch
##--enc_out_1x1_chs 1.0
## --dec_inp_chs 16 --dec_net_chs 16 
##--pretrained /home/lenovo/gxlong/work/ptlflow/rpknet,adam,100epochs,b8,lr0.0001/checkpoint.pth.tar 