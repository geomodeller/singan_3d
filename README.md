You may run this by 

'''
python main.py --input_name "3d_data_channel_(76, 88, 144).npy" --outdir "Output" --min_size 14 --niter 10_000 --nfc 32 --num_layer 3
'''
1. main.py : main script to train SinGAN
1. *.ipynb : to test the trained SinGAM to generate new realizations
1. *.npy   : 3D channelized reservoir (porosity but minmax scaler applied)
