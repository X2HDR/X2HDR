import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import imageio
import re
import cv2
from torchpercentile import Percentile

# For EXR files support
try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

# Update extensions to include HDR formats
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
HDR_EXTENSIONS = ['.hdr', '.exr']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def read_hdr_opencv(filepath):
    """Read HDR image using OpenCV (supports .hdr format)"""
    try:
        hdr_image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr_image is None:
            raise ValueError(f"Could not read HDR image: {filepath}")
        # Convert from BGR to RGB
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        return hdr_image
    except Exception as e:
        print(f"Error reading HDR with OpenCV: {e}")
        return None

def read_exr_openexr(filepath):
    """Read EXR image using OpenEXR library"""
    if not OPENEXR_AVAILABLE:
        return None
    
    try:
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channels = header['channels'].keys()
        
        if 'R' in channels and 'G' in channels and 'B' in channels:
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            
            r_str = exr_file.channel('R', FLOAT)
            g_str = exr_file.channel('G', FLOAT)
            b_str = exr_file.channel('B', FLOAT)
            
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
            
            hdr_image = np.stack([r, g, b], axis=2)
            return hdr_image
        else:
            print(f"RGB channels not found in {filepath}")
            return None
    except Exception as e:
        print(f"Error reading EXR: {e}")
        return None

def read_exr_opencv(filepath):
    """Read EXR image using OpenCV (alternative method)"""
    try:
        exr_image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if exr_image is None:
            raise ValueError(f"Could not read EXR image: {filepath}")
        # Convert from BGR to RGB
        exr_image = cv2.cvtColor(exr_image, cv2.COLOR_BGR2RGB)
        return exr_image
    except Exception as e:
        print(f"Error reading EXR with OpenCV: {e}")
        return None

def read_hdr_image(filepath):
    """Universal HDR image reader that handles both .hdr and .exr formats"""
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.hdr':
        return read_hdr_opencv(filepath)
    elif ext == '.exr':
        # Try OpenEXR first, fallback to OpenCV
        image = read_exr_openexr(filepath)
        if image is None:
            image = read_exr_opencv(filepath)
        return image
    else:
        # Fallback to imageio for other formats
        try:
            return imageio.imread(filepath)
        except Exception as e:
            print(f"Error reading image {filepath}: {e}")
            return None

def find_hdr_file(base_path, filename_without_ext):
    """Find HDR file with either .hdr or .exr extension"""
    for ext in HDR_EXTENSIONS:
        full_path = base_path + ext
        if os.path.exists(full_path):
            return full_path
    # If no HDR file found, return the original path for error handling
    return base_path + '.exr'

def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I

def image_seq_loader(img_seq_dir):
    img_seq_dir = os.path.expanduser(img_seq_dir)
    img_seq = []
    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                image_name = os.path.join(root, fname)
                im = np.array(Image.open(image_name).convert("RGB"))
                img_seq.append(torch.from_numpy(im/255).permute(2,0,1))
    return img_seq

def get_default_img_loader():
    return functools.partial(image_seq_loader)

class LDR_Seq(torch.nn.Module):
    def __init__(self):
        super(LDR_Seq, self).__init__()

    def get_luminance(self,img):
        if (img.shape[1] == 3):
            Y = img[:, 0, :, :] * 0.212656 + img[:, 1, :, :] * 0.715158 + img[:, 2, :, :] * 0.072186
        elif (img.shape[1] == 1):
            Y = img
        else:
            raise ValueError('get_luminance: wrong matrix dimension')
        return Y

    def get_weight(self, img):
        v_channel = self.get_luminance(img)
        gamma = 1 - 0.1
        gamma1 = 0.1
        v_channel[v_channel < gamma1] = 10**(-5)
        v_channel[v_channel > gamma] = 10**(-5)
        v_channel[v_channel > gamma1] = 1
        return v_channel

    def generation(self,img):
        img_q = img[img > 0]
        b = 1 / 128
        min_v = torch.min(img_q)
        img[img <= 0] = min_v
        L = self.get_luminance(img)
        img_l = torch.log2(L)
        l_img = Percentile()(img_l[:].reshape(1, -1).squeeze(), [0, 100])
        l_min = l_img[0]
        l_max = l_img[1]
        l_min = l_min
        f8_stops = torch.ceil((l_max - l_min) / 8)
        l_start = l_min
        number = 8 * 3 * f8_stops / 8
        number = number.long()

        result = []
        weight = []
        for i in range(number):
            k = (l_start + (8 / 3) * (i + 1))
            ek = 2 ** k
            img1 = (img / (ek + 0.00000001) - b) / (1 - b)
            imgClamp = img1.clamp(0.00000001, 1)
            imgP = (imgClamp) ** (1 / 2.2)
            all_len = len(imgP[imgP >= 0])
            white_len = len(imgP[imgP == 1])
            black_len = len(imgP[imgP <= 0.000232])
            pecent1 = white_len / all_len
            pecent2 = black_len / all_len
            if pecent1 < (3.5) / 4 and pecent2 <= 3 / 4:
                weight_temp = self.get_weight(imgP)
                result.append(imgP.squeeze())
                weight.append(weight_temp)
        return result,weight

class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 ref_dir,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            ref_dir (string): Directory of the reference images.
            test (bool): Whether this is test mode.
            get_loader: Image loader function.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None, dtype={0: str})
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.ref_dir = ref_dir
        self.test = test
        self.loader = get_loader()
        self.generate = LDR_Seq()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        filename_base = self.data.iloc[index, 0]
        
        # Find the actual image files (either .hdr or .exr)
        image_name = find_hdr_file(os.path.join(self.img_dir, filename_base), filename_base)
        image_ref_dir = find_hdr_file(os.path.join(self.ref_dir, filename_base), filename_base)
        
        # Read reference image
        ref_img_data = read_hdr_image(image_ref_dir)
        if ref_img_data is None:
            raise ValueError(f"Could not read reference image: {image_ref_dir}")
        
        img = torch.from_numpy(ref_img_data).permute(2, 0, 1).unsqueeze(0).float()
        imgref1, weight = self.generate.generation(img)
        imgref = torch.stack(imgref1, dim=0)
        
        # Read test image
        test_img_data = read_hdr_image(image_name)
        if test_img_data is None:
            raise ValueError(f"Could not read test image: {image_name}")
        
        # Normalize test image
        max_val = test_img_data.max()
        if max_val > 0:
            img = torch.from_numpy(test_img_data / max_val).permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(test_img_data).permute(2, 0, 1).float()
        
        sample = {'img': img, 'weight': weight, 'imgref': imgref, 'img_name': filename_base}
        return sample

    def __len__(self):
        return len(self.data.index)
