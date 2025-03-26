python inf.py rpknet  --way autocast --pyramid_ranges 4 2 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights --pretrained ./rpknet,adam,110epochs,b2,lr0.0001/checkpoint.pth.tar --data test_img --output ./test_img --enc_norm_type batch
## --plus
python test_img/cloud.py 
##python train_my.py rpknet --way autocast -b 2 --pyramid_ranges 4 2 --iters 12 --corr_mode allpairs --epochs 110 --not_cache_pkconv_weights --enc_norm_type batch
