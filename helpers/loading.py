import os
import numpy as np
import tqdm
import imageio
from helpers import coreutils
import tensorflow as tf


def discover_files(data_directory, n_images=120, v_images=30, extension='png', randomize=False):
    """
    Find available images and split them into training / validation sets.
    :param data_directory: directory
    :param n_images: number of training images
    :param v_images: number of validation images
    :param extension: file extension
    :param randomize: whether to shuffle files before the split
    """

    files = coreutils.listdir(data_directory, '.*\.{}$'.format(extension))
    print('In total {} files available'.format(len(files)), flush=True)

    if randomize:
        np.random.shuffle(files)

    if n_images == 0 and v_images == -1:
        v_images = len(files)

    if n_images == -1 and v_images == 0:
        n_images = len(files)

    if len(files) >= n_images + v_images:
        val_files = files[n_images:(n_images + v_images)]
        files = files[0:n_images]
    else:
        raise ValueError('Not enough images!')
        
    return files, val_files


def load_images(files, data_directory, extension='png', load='xy'):
    """
    Load pairs of full-resolution images: (raw, rgb). Raw inputs are stored in *.npy files (see
    train_prepare_training_set.py).
    :param files: list of files to be loaded
    :param data_directory: directory path
    :param extension: file extension of rgb images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    n_images = len(files)

    if n_images == 0:
        return {k: np.zeros(shape=(1, 1, 1, 1)) for k in load}
    
    # Check image resolution
    image = imageio.imread(os.path.join(data_directory, files[0]))
    resolutions = (image.shape[0] >> 1, image.shape[1] >> 1)
    del image
    
    data = {}
    
    if 'x' in load: data['x'] = np.zeros((n_images, *resolutions, 4), dtype=np.uint16)
    if 'y' in load: data['y'] = np.zeros((n_images, 2 * resolutions[0], 2 * resolutions[1], 3), dtype=np.uint8)

    with tqdm.tqdm(total=n_images, ncols=100, desc='Loading images') as pbar:

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')
            try:
                if 'x' in data: data['x'][i, :, :, :] = np.load(os.path.join(data_directory, npy_file))
                if 'y' in data: data['y'][i, :, :, :] = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')
            except Exception as e:
                print('Error: {} - {}'.format(file, e))
            pbar.update(1)

        return data

    
def load_patches(files, data_directory, patch_size=128, n_patches=100, discard_flat=False, extension='png', load='xy'):
    """
    Sample (raw, rgb) pairs or random patches from given images.
    :param files: list of available images
    :param data_directory: directory path
    :param patch_size: patch size (in the raw image - rgb patches will be twice as big)
    :param n_patches: number of patches per image
    :param discard_flat: remove flat patches
    :param extension: file extension of rgb images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    v_images = len(files)
    data = {}
    if 'x' in load: data['x'] = np.zeros((v_images * n_patches, patch_size, patch_size, 4), dtype=np.uint16)
    if 'y' in load: data['y'] = np.zeros((v_images * n_patches, 2 * patch_size, 2 * patch_size, 3), dtype=np.uint8)

    with tqdm.tqdm(total=v_images * n_patches, ncols=100, desc='Loading patches') as pbar:

        vpatch_id = 0

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')
            if 'x' in data: image_x = np.load(os.path.join(data_directory, npy_file))
            if 'y' in data: image_y = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')

            if 'x' in data:
                H, W = image_x.shape[0:2]
            elif 'y' in data:
                H, W = (x // 2 for x in image_y.shape[0:2])

            # Sample random patches
            panic_counter = 100 * n_patches

            for b in range(n_patches):
                found = False

                while not found: 
                    xx = np.random.randint(0, W - patch_size) if W - patch_size > 0 else 0
                    yy = np.random.randint(0, H - patch_size) if H - patch_size > 0 else 0
                    
                    if 'x' in data: data['x'][vpatch_id] = image_x[yy:yy + patch_size, xx:xx + patch_size, :]
                    if 'y' in data: data['y'][vpatch_id] = image_y[(2*yy):2*(yy + patch_size), (2*xx):2*(xx + patch_size), :]

                    # Check if the found patch is acceptable:
                    # - eliminate empty patches
                    if discard_flat and 'y' in data:
                        patch_variance = np.var(data['y'][vpatch_id])
                        if patch_variance < 1e-2:
                            panic_counter -= 1
                            found = False if panic_counter > 0 else True
                        elif patch_variance < 0.02:
                            found = np.random.uniform() > 0.5
                        else:
                            found = True
                    else:
                        found = True
                        
                vpatch_id += 1    
                pbar.update(1)

        return data

####### 改进效率

def load_images_dataset(files, data_directory, extension='png', load='xy'):
    """
    Load pairs of full-resolution images: (raw, rgb). Raw inputs are stored in *.npy files (see
    train_prepare_training_set.py).
    :param files: list of files to be loaded
    :param data_directory: directory path
    :param extension: file extension of rgb images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    n_images = len(files)

    if n_images == 0:
        return {k: np.zeros(shape=(1, 1, 1, 1)) for k in load}
    
    # Check image resolution
    image = imageio.imread(os.path.join(data_directory, files[0]))
    resolutions = (image.shape[0] >> 1, image.shape[1] >> 1)
    del image
    
    # data = {}
    
    # if 'x' in load: data['x'] = np.zeros((n_images, *resolutions, 4), dtype=np.uint16)
    # if 'y' in load: data['y'] = np.zeros((n_images, 2 * resolutions[0], 2 * resolutions[1], 3), dtype=np.uint8)

    npy_file = [file.replace('.{}'.format(extension), '.npy') for file in files]
    def _parse_function(raw_input_file, gt_file):	
        raw_input = tf.read_file(raw_input_file)
        gt_string = tf.read_file(gt_file)
        gt_decoded = tf.image.decode_image(gt_string)

        return raw_input, gt_decoded
    # 图片文件的列表
    raw_input_files = tf.constant(npy_file)
    # label[i]就是图片filenames[i]的label
    gt_files = tf.constant(files)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((raw_input_files, gt_files))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    # dataset = dataset.shuffle(buffersize=1000).batch().repeat(10)

    # def _gen():			
    #     X_train_r=np.zeros((n_images, *resolutions, 4), dtype=np.uint16)   #先构建一个框架出来，下面再赋值
    #     X_train_r[:,: ,0]=X_train[:,0:6000]     
    
    #     y_train=np.zeros((n_images, 2 * resolutions[0], 2 * resolutions[1], 3), dtype=np.uint8)
        
    #     leng=len(X_train_r)
    #     index=0
    #     while True:
    #         x_train_batch=X_train_r[index, :, 0:1]
    #         y_train_batch=y_train[index, :]
    #         yield (x_train_batch, y_train_batch)
    #         index=index+1
    #         if index>leng:
    #             break
    # with tqdm.tqdm(total=n_images, ncols=100, desc='Loading images') as pbar:

    #     for i, file in enumerate(files):
    #         npy_file = file.replace('.{}'.format(extension), '.npy')
    #         try:
    #             if 'x' in data: data['x'][i, :, :, :] = np.load(os.path.join(data_directory, npy_file))
    #             if 'y' in data: data['y'][i, :, :, :] = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')
    #         except Exception as e:
    #             print('Error: {} - {}'.format(file, e))
    #         pbar.update(1)

    return dataset

    
def load_patches_dataset(files, data_directory, patch_size=128, n_patches=100, discard_flat=False, extension='png', load='xy'):
    """
    Sample (raw, rgb) pairs or random patches from given images.
    :param files: list of available images
    :param data_directory: directory path
    :param patch_size: patch size (in the raw image - rgb patches will be twice as big)
    :param n_patches: number of patches per image
    :param discard_flat: remove flat patches
    :param extension: file extension of rgb images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    v_images = len(files)
    # data = {}
    # if 'x' in load: data['x'] = np.zeros((v_images * n_patches, patch_size, patch_size, 4), dtype=np.uint16)
    # if 'y' in load: data['y'] = np.zeros((v_images * n_patches, 2 * patch_size, 2 * patch_size, 3), dtype=np.uint8)

    npy_file = [file.replace('.{}'.format(extension), '.npy') for file in files]
    def _parse_function(raw_input_file, gt_file):	
        raw_input = tf.read_file(raw_input_file)
        gt_string = tf.read_file(gt_file)
        gt_decoded = tf.image.decode_image(gt_string)

        return raw_input, gt_decoded
    # 图片文件的列表
    raw_input_files = tf.constant(npy_file)
    # label[i]就是图片filenames[i]的label
    gt_files = tf.constant(files)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((raw_input_files, gt_files))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # with tqdm.tqdm(total=v_images * n_patches, ncols=100, desc='Loading patches') as pbar:

    #     vpatch_id = 0

    #     for i, file in enumerate(files):
    #         npy_file = file.replace('.{}'.format(extension), '.npy')
    #         if 'x' in data: image_x = np.load(os.path.join(data_directory, npy_file))
    #         if 'y' in data: image_y = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')

    #         if 'x' in data:
    #             H, W = image_x.shape[0:2]
    #         elif 'y' in data:
    #             H, W = (x // 2 for x in image_y.shape[0:2])

    #         # Sample random patches
    #         panic_counter = 100 * n_patches

    #         for b in range(n_patches):
    #             found = False

    #             while not found: 
    #                 xx = np.random.randint(0, W - patch_size) if W - patch_size > 0 else 0
    #                 yy = np.random.randint(0, H - patch_size) if H - patch_size > 0 else 0
                    
    #                 if 'x' in data: data['x'][vpatch_id] = image_x[yy:yy + patch_size, xx:xx + patch_size, :]
    #                 if 'y' in data: data['y'][vpatch_id] = image_y[(2*yy):2*(yy + patch_size), (2*xx):2*(xx + patch_size), :]

    #                 # Check if the found patch is acceptable:
    #                 # - eliminate empty patches
    #                 if discard_flat and 'y' in data:
    #                     patch_variance = np.var(data['y'][vpatch_id])
    #                     if patch_variance < 1e-2:
    #                         panic_counter -= 1
    #                         found = False if panic_counter > 0 else True
    #                     elif patch_variance < 0.02:
    #                         found = np.random.uniform() > 0.5
    #                     else:
    #                         found = True
    #                 else:
    #                     found = True
                        
    #             vpatch_id += 1    
    #             pbar.update(1)

    return dataset