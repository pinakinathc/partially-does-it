''' This dataloader should work for both SketchyScene and SketchyCOCO.
First, we need to filter these noisy datasets following instructions mentioned in each paper.

-------------------------------------------------
Some Quantitative values claimed in their paper.
-------------------------------------------------
* SketchyScene -- 32.13% Acc@1 | 69.48% Acc@10

* SketchyCOCO -- 31.91% Acc@1 | 86.19% Acc@10

None of these papers have released their code or the sequences of images for which they got this result.

'''

import os
import glob
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch

class SketchyScene(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        self.sketch_dir = os.path.join(self.opt.root_dir, mode, 'INSTANCE_GT')
        self.image_dir = os.path.join(self.opt.root_dir, mode, 'reference_image')

        self.list_ids = self.filter(self.sketch_dir, self.image_dir)

    def filter(self, sketch_dir, image_dir):
        ''' Images and Sketches have some inconsistency, hence filtering is required -- although heuristic '''
        list_sk_ids = [int(os.path.split(x)[-1].split('_')[1]) for x in glob.glob(os.path.join(sketch_dir, '*.mat'))]
        list_img_ids = [int(os.path.split(x)[-1][:-4]) for x in glob.glob(os.path.join(image_dir, '*.jpg'))]

        return [x for x in list_sk_ids if x in list_img_ids] # intersection

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        index = self.list_ids[index]
        sketch_data = loadmat(os.path.join(self.sketch_dir, 'sample_%d_instance.mat'%index))['INSTANCE_GT']
        image_data = Image.open(os.path.join(self.image_dir, '%d.jpg'%index))
        negative_data = Image.open(os.path.join(self.image_dir, '%d.jpg'%np.random.choice(self.list_ids, 1)[0]))
        
        # Partial data
        sketch_data = self.partial_data(sketch_data, p_mask=self.opt.p_mask)
        sketch_data = Image.fromarray(sketch_data).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            img_tensor = self.transform(image_data)
            sk_tensor = self.transform(sketch_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor

    def partial_data(self, sketch_data, p_mask):
        partial_sketch = np.zeros_like(sketch_data)
        instances = np.unique(sketch_data)[1:] # Remove 0-th element
        selected_instances = np.random.choice(instances, round(len(instances)*(1-p_mask)), replace=False)
        for obj in selected_instances:
            # if np.random.random_sample() > p_mask:
            partial_sketch[sketch_data == obj] = 255
        return partial_sketch


class SketchyCOCO(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        if mode == 'train':
            self.mode = 'trainInTrain'
        else:
            self.mode = 'val'

        self.all_ids = glob.glob(os.path.join(
            self.opt.root_dir, 'Annotation', 'paper_version', self.mode, 'CLASS_GT', '*.mat'))

        self.all_ids = [os.path.split(filepath)[-1][:-4] 
            for filepath in self.all_ids if (np.unique(loadmat(filepath)['CLASS_GT']) <16).sum() >= 2]
        print ('total %s samples: %d'%(self.mode, len(self.all_ids)))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]
        # sketch_file = os.path.join(self.opt.root_dir, 'Sketch', 'paper_version', self.mode, '%s.png'%filename)
        sketch_data = loadmat(os.path.join(self.opt.root_dir, 'Annotation', 'paper_version', self.mode, 'INSTANCE_GT', '%s.mat'%filename))['INSTANCE_GT']
        image_file = os.path.join(self.opt.root_dir, 'GT', self.mode, '%s.png'%filename)
        negative_file = os.path.join(self.opt.root_dir, 'GT', self.mode, '%s.png'%np.random.choice(self.all_ids, 1)[0])

        # sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        # Partial data
        sketch_data = self.partial_data(sketch_data, p_mask=self.opt.p_mask)
        sketch_data = Image.fromarray(sketch_data).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor

    def partial_data(self, sketch_data, p_mask):
        partial_sketch = np.zeros_like(sketch_data)
        instances = np.unique(sketch_data)[1:] # Remove 0-th element
        # print ('number of elements selected: ', round(len(instances)*(1-p_mask)))
        selected_instances = np.random.choice(instances, round(len(instances)*(1-p_mask)), replace=False)
        # print (selected_instances)
        for obj in selected_instances:
            # if np.random.random_sample() > p_mask:
            partial_sketch[sketch_data == obj] = 255
        return partial_sketch


class PhotoSketching(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        self.image_dir = os.path.join(self.opt.root_dir, 'image')
        self.sketch_dir = os.path.join(self.opt.root_dir, 'sketch-rendered', 'width-3')
        if mode == 'train':
            self.all_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'list', 'train.txt'), dtype=str)
        else:
            self.all_ids = list(np.loadtxt(os.path.join(self.opt.root_dir, 'list', 'val.txt'), dtype=str))
            self.all_ids.extend(list(np.loadtxt(os.path.join(self.opt.root_dir, 'list', 'test.txt'), dtype=str)))
        print ('total data: ', len(self.all_ids))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        idx = self.all_ids[index]
        sketch_data = Image.open(os.path.join(self.sketch_dir, '%s_03.png'%idx)).convert('RGB')
        image_data = Image.open(os.path.join(self.image_dir, '%s.jpg'%idx)).convert('RGB')
        negative_data = Image.open(os.path.join(self.image_dir, '%s.jpg'%np.random.choice(self.all_ids, 1)[0])).convert('RGB')

        # Partial data
        sketch_data = self.partial_data(sketch_data, p_mask=self.opt.p_mask)
        sketch_data = Image.fromarray(sketch_data).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            img_tensor = self.transform(image_data)
            sk_tensor = self.transform(sketch_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor

    def partial_data(self, sketch_data, p_mask):
        sketch_data = np.array(sketch_data, dtype=np.uint8)
        H, W = sketch_data.shape[:2]
        del_h = int(H*(p_mask**0.5))
        del_w = int(W*(p_mask**0.5))
        h = int(np.random.uniform(0, H-del_h))
        w = int(np.random.uniform(0, W-del_w))
        sketch_data[h:h+del_h, w:w+del_w, :] = 255
        return sketch_data


class Shoev2(torch.utils.data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.base_dir, hp.root_dir, 'Dataset', hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.base_dir, hp.root_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x]
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            possible_list = list(range(len(self.Train_Sketch)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)]
            negative_sample = '_'.join(self.Train_Sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample,
                       'sketch_points':sketch_points
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB'))

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'sketch_points': sketch_points}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


if __name__ == '__main__':
    from src.deepemd.options import opts
    from torchvision import transforms

    output_dir = 'output_data'
    os.makedirs(output_dir, exist_ok=True)

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PhotoSketching(opts, mode='train', transform=dataset_transforms, return_orig=True)
    for iter_ in range(1):
        for idx, (sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data) in enumerate(dataset):
            sketch_data.save(os.path.join(output_dir, '%d_%d_sk.jpg'%(idx, iter_)))
            image_data.save(os.path.join(output_dir, '%d_%d_img.jpg'%(idx, iter_)))
            negative_data.save(os.path.join(output_dir, '%d_%d_neg.jpg'%(idx, iter_)))
            # if idx > 7:
            #     break


            print ('Shape of sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
                sk_tensor.shape, img_tensor.shape, neg_tensor.shape))
