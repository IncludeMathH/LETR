"""
LETR Basic Usage Demo
"""
import torch, os, argparse, glob
import cv2
from tqdm import tqdm

import torchvision.transforms.functional as functional
import torch.nn.functional as F
from models import build_model
from util.misc import nested_tensor_from_tensor_list


def parse_args():
    parser = argparse.ArgumentParser(description='LETR Basic Usage Demo Setup')
    parser.add_argument('--img_dir', help='the dir of input image',
                        default='data/wireframe_processed/val2017/00031546.png')
    parser.add_argument('--output_dir', help='the dir to save processed image',
                        default='data/inference_figures')
    parser.add_argument('--ckpt_dir', help='the checkpoint dir to be used',
                        default='exp/res101_stage2_focal/checkpoints/checkpoint0024.pth')
    args = parser.parse_args()

    return args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image

class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)

def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image

class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)

def post_processing(outputs, orig_size):
    # Post-processing Results
    out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    img_h, img_w = orig_size.unbind(0)
    scale_fct = torch.unsqueeze(torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line * scale_fct[:, None, :]
    lines = lines.view(1000, 2, 2)
    lines = lines.flip([-1])# this is yxyx format
    scores = scores.detach().numpy()
    keep = scores >= 0.7
    keep = keep.squeeze()
    lines = lines[keep]
    if lines.numel() == 0:
        return []
    lines = lines.reshape(lines.shape[0], -1)
    return lines

def visual_save(img_dir, lines, output_dir, output_img_name):
    # Plot Inference Results
    input_image = cv2.imread(img_dir)        
    for tp_id, line in enumerate(lines):
        y1, x1, y2, x2 = line # this is yxyx
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.line(input_image, p1, p2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, output_img_name), img=input_image)

def main():
    args = parse_args()
    # 指定输入图像
    imgs_dir = args.img_dir
    output_dir = args.output_dir
    # obtain checkpoints
    checkpoint = torch.load(args.ckpt_dir, map_location='cpu')

    try:
        os.makedirs(output_dir)
    except OSError:
        print(f'{output_dir} already exists')
    else:
        print(f'create folder in {output_dir}')
    
    # load model
    args = checkpoint['args']
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # get image name and generate output image name
    if os.path.isfile(imgs_dir):
        imgs_dir = [imgs_dir]
    elif os.path.isdir(imgs_dir):
        imgs_dir = glob.glob(os.path.join(imgs_dir, '*'))
    
    for img_dir in tqdm(imgs_dir):
        img_name = os.path.basename(img_dir)
        output_img_name = 'res_' + img_name

        # load image
        raw_img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        h, w = raw_img.shape[0], raw_img.shape[1]
        orig_size = torch.as_tensor([int(h), int(w)])

        # normalize image
        test_size = 1100
        normalize = Compose([
                ToTensor(),
                Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
                Resize([test_size]),
            ])
        img = normalize(raw_img)
        inputs = nested_tensor_from_tensor_list([img])

        # Model Inference
        with torch.no_grad():
            outputs = model(inputs)[0]

        lines = post_processing(outputs=outputs, orig_size=orig_size)
        visual_save(img_dir, lines, output_dir, output_img_name)

    print('output image has been saved!')


if __name__ == '__main__':
    main()
