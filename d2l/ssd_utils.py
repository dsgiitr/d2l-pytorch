import os
import errno
from tqdm import tqdm
import torch
import json
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import sys
d2l = sys.modules[__name__]

# Defined in file: ./chapter_preface/preface.md
from matplotlib import pyplot as plt

import d2l
import json
import time
from collections import namedtuple
import cv2
from IPython import display


##################################### Display Functions #################################################

# Defined in file: ./chapter_crashcourse/probability.md
def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    display.set_matplotlib_formats('svg')


# Defined in file: ./chapter_crashcourse/probability.md
def set_figsize(figsize=(3.5, 2.5)):
    """Change the default figure size"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# Defined in file: ./chapter_crashcourse/naive-bayes.md
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def read_img(img_str: str, target_size: int) -> np.ndarray:
    img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return img


def draw_boxes(img: str, boxes: list) -> np.ndarray:
    for box in boxes:
        cv2.rectangle(img, (int(box[0] - box[2]/2), int(box[1] - box[3]/2)),
                      (int(box[0] + box[2]/2), int(box[1] + box[3]/2)),(0, 0, 255), 2)
    return img


def draw_grid(img: str, pixel_step: int) -> np.ndarray:
    x = pixel_step
    y = pixel_step

    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 255, 255))
        x += pixel_step

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 255, 255))
        y += pixel_step

    return img


def draw_text(img: str, texts: list, locations: list) -> np.ndarray:
    for text, loc in zip(texts, locations):
        cv2.putText(img, text, (int(loc[0]-(loc[2]/2)-5), int(loc[1]-(loc[3]/2)-5)), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 0, 0), 1)
    return img


################################ Functions for making an inference on an image using trained model ###########################

PredBoundingBox = namedtuple("PredBoundingBox", ["probability", "class_id",
                                                 "classname", "bounding_box"
                                                 ])

def invert_transformation(bb_hat, anchors):
    """
    Invert the transform from "loc_transformation".
    """

    return torch.stack([anchors[:, 0] + bb_hat[:, 0] * anchors[:, 2],
                        anchors[:, 1] + bb_hat[:, 1] * anchors[:, 3],
                        anchors[:, 2] * torch.exp(bb_hat[:, 2]),
                        anchors[:, 3] * torch.exp(bb_hat[:, 3])
                        ], dim=1)

def non_max_suppression(bounding_boxes: list, iou_threshold: float = 0.1) -> list:
    filtered_bb = []

    while len(bounding_boxes) != 0:
        best_bb = bounding_boxes.pop(0)
        filtered_bb.append(best_bb)

        remove_items = []
        for bb in bounding_boxes:
            iou = jaccard(torch.tensor(best_bb.bounding_box).unsqueeze(0), 
                          torch.tensor(bb.bounding_box).unsqueeze(0))

            if iou > iou_threshold:
                remove_items.append(bb)
        bounding_boxes = [bb for bb in bounding_boxes if bb not in remove_items]
    return filtered_bb


def infer(net, epoch, background_threshold=0.9, device = "cuda:0"):
    
    img = np.array(Image.open('../img/pikachu.jpg').convert('RGB').resize((256, 256), Image.BILINEAR))
    X = transforms.Compose([transforms.ToTensor()])(img).to(device)
    
    X = X.to(device)
    
    # background_threshold = 0.9

    net.eval()
    anchors, class_hat, bb_hat = net(X.unsqueeze(0))

    anchors = anchors.to(device)


    bb_hat = bb_hat.reshape((1, -1, 4))

    bb_hat = invert_transformation(bb_hat.squeeze(0), anchors)
    bb_hat = bb_hat * 256.0

    class_hat = class_hat.sigmoid().squeeze(0)

    bb_hat = bb_hat[class_hat[:,0] < background_threshold, :]


    bb_hat = bb_hat.detach().cpu().numpy()
    class_hat = class_hat[class_hat[:,0] < background_threshold, :]

    class_preds = class_hat[:, 1:]

    prob, class_id = torch.max(class_preds,1)

    prob = prob.detach().cpu().numpy()
    class_id = class_id.detach().cpu().numpy()


    id_cat_pikachu = dict()
    id_cat_pikachu[0] = 'pikachu'
    
    output_bb = [PredBoundingBox(probability=prob[i],
                                 class_id=class_id[i],
                                 classname=id_cat_pikachu[class_id[i]],
                                 bounding_box=[bb_hat[i, 0], 
                                               bb_hat[i, 1], 
                                               bb_hat[i, 2], 
                                               bb_hat[i, 3]])
                                 for i in range(0, len(prob))]

    output_bb = sorted(output_bb, key = lambda x: x.probability, reverse=True)

    filtered_bb = non_max_suppression(output_bb)

    img_str = '../img/pikachu.jpg'
    img = read_img(img_str, 256)

    # img = (X.cpu().numpy().transpose(1,2,0)*255)
    # img1 = np.ascontiguousarray(img, dtype=np.uint8)

    img = draw_boxes(img, [bb.bounding_box for bb in filtered_bb])
    img = draw_text(img, [bb.classname for bb in filtered_bb], [bb.bounding_box for bb in filtered_bb])
    plt.imsave('ssd_outputs/img_' + str(epoch) + '.png', img)


########################################## Functions for downloading and preprocessing data ##############################################


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e

# The following code is used to download pikachu dataset from mxnet aws server in the form of .rec files
# Then this .rec file is converted into png images and json files for annotation data
# This part requires 'mxnet' library which can be downloaded using conda 
# using the command 'conda install mxnet'
# Matplotlib is also required for saving the png image files

# Download Pikachu Dataset
def download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        download_url(root_url + k, data_dir)


# Create dataloaders in mxnet
def load_data_pikachu_rec_mxnet(batch_size, edge_size=256):
    from mxnet import image
    """Load the pikachu dataset"""
    data_dir = '../data/pikachu'
    download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # The shape of the output image
        # shuffle=True,  # Read the data set in random order
        # rand_crop=1,  # The probability of random cropping is 1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter


# Use mxnet dataloaders to convert .rec file to .png images and annotations.json

def download_and_preprocess_data(dir = '../data/pikachu/'):
    
    if os.path.exists(os.path.join(dir, 'train')) and os.path.exists(os.path.join(dir, 'val')):
        return

    train_iter, val_iter = load_data_pikachu_rec_mxnet(batch_size)
    
    os.mkdir(os.path.join(dir, 'train'))
    os.mkdir(os.path.join(dir, 'val'))
    os.mkdir(os.path.join(dir, 'train/images'))
    os.mkdir(os.path.join(dir, 'val/images'))
    
    annotations_train = dict()
    train_iter.reset()  # Read data from the start.
    id = 0
    for batch in train_iter:
        id+=1
        
        X = batch.data[0].as_in_context(ctx)
    
        Y = batch.label[0].as_in_context(ctx)

        x = X.asnumpy()
        x = x.transpose((2,3,1,0))
        x = x.squeeze(axis=-1)
        plt.imsave(os.path.join(dir, 'train/images', 'pikachu_' + str(id) + '.png'), x/255.)
        an = dict()
        y = Y.asnumpy()
    
        an['class'] = y[0, 0][0].tolist()
        an['loc'] = y[0,0][1:].tolist()
        an['id'] = [id]
        an['image'] = 'pikachu_' + str(id) + '.png'
        annotations_train['data_' + str(id)] = an 

    import json
    with open(os.path.join(dir, 'train', 'annotations.json'), 'w') as outfile:
        json.dump(annotations_train, outfile)
    outfile.close()


    annotations_val = dict()
    val_iter.reset()  # Read data from the start.
    id = 0
    for batch in val_iter:
        id+=1
        
        X = batch.data[0].as_in_context(ctx)
    
        Y = batch.label[0].as_in_context(ctx)

        x = X.asnumpy()
        x = x.transpose((2,3,1,0))
        x = x.squeeze(axis=-1)
        plt.imsave(os.path.join(dir, 'val/images', 'pikachu_' + str(id) + '.png'), x/255.)
        an = dict()
        y = Y.asnumpy()
    
        an['class'] = y[0, 0][0].tolist()
        an['loc'] = y[0,0][1:].tolist()
        an['id'] = [id]
        an['image'] = 'pikachu_' + str(id) + '.png'
        annotations_val['data_' + str(id)] = an 

    import json
    with open(os.path.join(dir, 'val', 'annotations.json'), 'w') as outfile:
        json.dump(annotations_val, outfile)
    outfile.close()


################################################## PyTorch Dataloader for PIKACHU dataset ########################################################

class PIKACHU(torch.utils.data.Dataset):
    def __init__(self, data_dir, set, transform=None, target_transform=None):
        
        self.image_size = (3, 256, 256)
        self.images_dir = os.path.join(data_dir, set, 'images')

        self.set = set
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.target_transform = target_transform

        annotations_file = os.path.join(data_dir, set, 'annotations.json')
        with open(annotations_file) as file:
            self.annotations = json.load(file)

    def __getitem__(self, index):
        
        annotations_i = self.annotations['data_' + str(index+1)]
        
        image_path = os.path.join(self.images_dir, annotations_i['image'])
        img = np.array(Image.open(image_path).convert('RGB').resize((self.image_size[2], self.image_size[1]), Image.BILINEAR))
        # print(img.shape)
        loc = np.array(annotations_i['loc'])
        
        loc_chw = np.zeros((4,))
        loc_chw[0] = (loc[0] + loc[2])/2
        loc_chw[1] = (loc[1] + loc[3])/2
        loc_chw[2] = (loc[2] - loc[0])  #width
        loc_chw[3] = (loc[3] - loc[1])  # height
        

        label = 1 - annotations_i['class']
    
        if self.transform is not None:
            img = self.transform(img)       
        return (img, loc_chw, label)

    def __len__(self):
        return len(self.annotations)

###################################################### Functions for showing graph of loss function ##############################################

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()

class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


################################## Functions for finding the overlap between anchors and predicted bounding boxes using IoU #######################

def center_2_hw(box: torch.Tensor) -> float:
    """
    Converting (cx, cy, w, h) to (x1, y1, x2, y2)
    """

    return torch.cat(
        [box[:, 0, None] - box[:, 2, None]/2,
         box[:, 1, None] - box[:, 3, None]/2,
         box[:, 0, None] + box[:, 2, None]/2,
         box[:, 1, None] + box[:, 3, None]/2
         ], dim=1)

def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    # Coverting (cx, cy, w, h) to (x1, y1, x2, y2) since its easier to extract min/max coordinates
    temp_box_a, temp_box_b = center_2_hw(box_a), center_2_hw(box_b)
    
#     print(temp_box_a.shape)
    
    max_xy = torch.min(temp_box_a[:, None, 2:], temp_box_b[None, :, 2:])
    min_xy = torch.max(temp_box_a[:, None, :2], temp_box_b[None, :, :2])
    
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def box_area(box: torch.Tensor) -> float:
    return box[:, 2] * box[:, 3]

def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
#     print(box_a.shape)
    intersection = intersect(box_a, box_b)
    union = box_area(box_a).unsqueeze(1) + box_area(box_b).unsqueeze(0) - intersection
    return intersection / union

def find_overlap(bb_true_i, anchors, jaccard_overlap):

    jaccard_tensor = jaccard(anchors, bb_true_i)
    _, max_overlap = torch.max(jaccard_tensor, dim=0)
    
    overlap_list = []    
    for i in range(len(bb_true_i)):
        threshold_overlap = (jaccard_tensor[:, i] > jaccard_overlap).nonzero()

        if len(threshold_overlap) > 0:
            threshold_overlap = threshold_overlap[:, 0]
            overlap = torch.cat([max_overlap[i].view(1), threshold_overlap])
            overlap = torch.unique(overlap)     
        else:
            overlap = max_overlap[i].view(1)
        overlap_list.append(overlap)
    return overlap_list


#################################### Functions for saving and loading trained models #############################################

def save(model, path_to_checkpoints_dir, step, optimizer, loss):
    
    try:
        os.makedirs(path_to_checkpoints_dir)
    except:
        pass

    path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}_{loss}.pth')
    checkpoint = {
        'state_dict': model.state_dict(),
        'step': step,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint

def load(model, path_to_checkpoint, optimizer):
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    step = checkpoint['step']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return step


############################ Object Detection Related Functions ##############################




import itertools
import math
def MultiBoxPrior(feature_map_sizes, sizes, aspect_ratios):
    """Compute default box sizes with scale and aspect transform."""
    
    sizes = [s*728 for s in sizes]
    
    scale = feature_map_sizes
    steps_y = [1 / scale[0]]
    steps_x = [1 / scale[1]]
    
    sizes = [s / max(scale) for s in sizes]
    
    num_layers = 1

    boxes = []
    for i in range(num_layers):
        for h, w in itertools.product(range(feature_map_sizes[0]), range(feature_map_sizes[1])):
            cx = (w + 0.5)*steps_x[i]
            cy = (h + 0.5)*steps_y[i]
            
            for j in range(len(sizes)):

                s = sizes[j]
                boxes.append((cx, cy, s, s))

            s = sizes[0]
            
            for ar in aspect_ratios:
               
                boxes.append((cx, cy, (s * math.sqrt(ar)), (s / math.sqrt(ar))))

    return torch.Tensor(boxes) 

def MultiBoxTarget(class_true, bb_true, anchors):
    
    class_true +=1
    
    class_target = torch.zeros(anchors.shape[0]).long()

    overlap_list = d2l.find_overlap(bb_true, anchors, 0.5)
    
    overlap_coordinates = torch.zeros_like(anchors)
    
    for j in range(len(overlap_list)):
        overlap = overlap_list[j]
        class_target[overlap] = class_true[j, 0]
        overlap_coordinates[overlap] = 1.
        
        
    
    new_anchors = torch.cat([*anchors])
    overlap_coordinates = torch.cat([*overlap_coordinates])
    new_anchors = new_anchors*overlap_coordinates
    
    return (new_anchors.unsqueeze(0), overlap_coordinates.unsqueeze(0), class_target.unsqueeze(0))


def MultiboxDetection(id_cat, cls_probs, anchors, nms_threshold):

    id_new = dict()
    id_new[0] = 'background'
    for i in (id_cat.keys()):
        id_new[i+1] = id_cat[i]

    cls_probs = cls_probs.transpose(0,1)

    prob, class_id = torch.max(cls_probs,1)

    prob = prob.detach().cpu().numpy()
    class_id = class_id.detach().cpu().numpy()

    output_bb = [d2l.PredBoundingBox(probability=prob[i],
                                     class_id=class_id[i],
                                     classname=id_new[class_id[i]],
                                     bounding_box=[anchors[i, 0], 
                                                   anchors[i, 1], 
                                                   anchors[i, 2], 
                                                   anchors[i, 3]])
                                     for i in range(0, len(prob))]

    filtered_bb = d2l.non_max_suppression(output_bb, nms_threshold)
    
    out = []
    for bb in filtered_bb:
        out.append([bb.class_id-1, bb.probability, *bb.bounding_box])
    out = torch.Tensor(out)
    
    return out

############################ Functions for Multi-Scale Object Detection Jupyter Notebook ##########################

# def bbox_to_rect(bbox, color):
#     """Convert bounding box to matplotlib format."""
#     return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
#                          height=bbox[3]-bbox[1], fill=False, edgecolor=color,
#                          linewidth=2)

# def show_bboxes(axes, bboxes, labels=None, colors=None):
#     """Show bounding boxes."""
#     def _make_list(obj, default_values=None):
#         if obj is None:
#             obj = default_values
#         elif not isinstance(obj, (list, tuple)):
#             obj = [obj]
#         return obj
#     labels = _make_list(labels)
#     colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
#     for i, bbox in enumerate(bboxes):
#         color = colors[i % len(colors)]
#         rect = bbox_to_rect(bbox.numpy(), color)
#         axes.add_patch(rect)
#         if labels and len(labels) > i:
#             text_color = 'k' if color == 'w' else 'w'
#             axes.text(rect.xy[0], rect.xy[1], labels[i],
#                       va='center', ha='center', fontsize=9, color=text_color,
#                       bbox=dict(facecolor=color, lw=0))

def get_centers(h, w, fh, fw):
    step_x = int(w/fw)
    cx = []
    for i in range(fw):
        cx.append((step_x*i + step_x*(i+1))/2)
    
    step_y = int(h/fh)
    cy = []
    for j in range(fh):
        cy.append((step_y*j + step_y*(j+1))/2)
    cxcy = []
    for x in cx:
        for y in cy:
            cxcy.append([x, y])
    
    return np.array(cxcy).astype(np.int16)

####################################################################################################################################
