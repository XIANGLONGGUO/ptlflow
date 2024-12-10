import argparse
from path import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# import models
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from imageio import imread, imwrite
import numpy as np
from ptlflow import get_model, get_model_reference
from ptlflow.utils.callbacks.logger import LoggerCallback
from ptlflow.utils.utils import (
    add_datasets_to_parser,
    get_list_of_available_models_list,
)


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__"))
def get():
    parser = argparse.ArgumentParser(description='StrainNet inference',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model',
                        help='network f or h')                                  
    parser.add_argument('--data', metavar='DIR',
                        help='path to images folder, image names must match \'[name]1.[ext]\' and \'[name]2.[ext]\'')
    parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
    parser.add_argument('--output', '-o', metavar='DIR', default=None,
                        help='path to output folder. If not set, will be created in data folder')
    parser.add_argument('--div-flow', default=2, type=float,
                        help='value by which flow will be divided')
    parser.add_argument("--img-exts", metavar='EXT', default=['tif','png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                        help="images extensions to glob")
    return parser
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    import sys
    global args, best_EPE
    parser = get()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None

    FlowModel = get_model_reference(sys.argv[1])
    parser = FlowModel.add_model_specific_args(parser)
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()
    
    # Data loading code
    input_transform = transforms.Compose([transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ])
    import os
    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*1.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.stem[:-1] + '2.{}'.format(ext))
            if os.path.isfile(img_pair):
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.model))
    model = get_model(args.model, args.pretrained,args).to(device)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    cudnn.benchmark = True


    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 =  np.array(imread(img1_file))
        img2 =  np.array(imread(img2_file))
        
        img1 = img1/255
        img2 = img2/255
            
        if img1.ndim == 2:         
            img1 = img1[np.newaxis, ...]       
            img2 = img2[np.newaxis, ...]
        
            img1 = img1[np.newaxis, ...]       
            img2 = img2[np.newaxis, ...]
            
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()       

            img1 = torch.cat([img1,img1,img1],1)
            img2 = torch.cat([img2,img2,img2],1)
            input_var = torch.cat([img1,img2],1)           

        elif img1.ndim == 3:
            img1 = np.transpose(img1, (2, 0, 1))
            img2 = np.transpose(img2, (2, 0, 1))        
        
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()       
            input_var = torch.cat([img1, img2]).unsqueeze(0)          
        input={}
        input['images'] = torch.stack([img1, img2], dim=1) 
        # compute output   
        # input_var = input_var.to(device)
        output = model(input)
        output = output['flows'].squeeze(1)
        output_to_write = output.data.cpu()
        output_to_write = output_to_write.numpy()       
        disp_x = output_to_write[0,0,:,:]
        disp_x = - disp_x * args.div_flow + 1        
        disp_y = output_to_write[0,1,:,:]
        disp_y = - disp_y * args.div_flow + 1

        filenamex = save_path/'{}{}'.format(img1_file.stem[:-1], '_disp_x')
        filenamey = save_path/'{}{}'.format(img1_file.stem[:-1], '_disp_y')        
        np.savetxt(filenamex + '.csv', disp_x,delimiter=',')
        np.savetxt(filenamey + '.csv', disp_y,delimiter=',')
        
   
if __name__ == '__main__':
    main()
    #python inf.py rpknet --pyramid_ranges 32 8 --iters 12 --corr_mode allpairs --not_cache_pkconv_weights --pretrained ./rpknet,adam,300epochs,b8,lr0.0001/checkpoint.pth.tar --data test_img --output ./test_img

