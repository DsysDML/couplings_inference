import numpy as np
import h5py
import os

# script to cleaning model files

model_fname = 'PottsBernoulliRBM_PCD-10_mbs=5000_lr=0.01_Nh=1000_centered_old.h5'

updates_2_save = np.array([1080, 3780, 14580, 51480, 205920, 454140])

def list_ends_with(string, token_list):
    for token in token_list:
        if string.endswith('W' + str(int(token))) or string.endswith('s' + str(int(token))):
            return True
    return False

# strainer
keys_list = []
i = 0
with h5py.File(f'models/{model_fname}', 'r+') as f:
    keys_list = list(f.keys())
    keys_list.remove('UpdByEpoch')
    strain_word = 'X_pc'
    n_upd = f['UpdByEpoch'][()]
    print(n_upd)
    for key in keys_list:
        if key.startswith(strain_word):
            del f[key]    
        elif not list_ends_with(key, updates_2_save/n_upd):
            del f[key]
    keys_list = list(f.keys())
    
# creating new file
original_file = f'./models/{model_fname}'
temp_file = './models/PottsBernoulliRBM_PCD-10_mbs=5000_lr=0.01_Nh=1000_centered.h5'

with h5py.File(original_file, 'r') as f_original, h5py.File(temp_file, 'w') as f_temp:
    def copy_items(name, obj):
        if isinstance(obj, h5py.Dataset):
            if obj.shape == ():  
                f_temp.create_dataset(name, data=obj[()])
            else:
                f_temp.create_dataset(name, data=obj[:])
        elif isinstance(obj, h5py.Group):
            f_temp.create_group(name)

    f_original.visititems(copy_items)

# reeplacing the original file with the temp file
os.replace(temp_file, original_file)