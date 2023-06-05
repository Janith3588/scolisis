import argparse
import train
import test
import eval
from logzero import logger
import sys
from pathlib import Path
import json
import numpy as np
import skimage
import skimage.io
import skimage.transform
import torch
from PIL import Image
import cobb_draw
from logzero import logger
from utils import (create_model, load_image, resize_height,
                   normalize_intensity, pad_to_shape, apply_clahe,
                   base_transform, heatmap2rgb, heatmap2points, rgb_on_gray)
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=100, help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_last.pth', help='weights to be resumed')
    parser.add_argument('--data_dir', type=str, default='Datasets\spinal', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    parser.add_argument('input', help='Input filename:', metavar='<input>')
    parser.add_argument('output', help='Output filename', metavar='<output>')
    parser.add_argument('-w',
                        '--weights',
                        help='pt file path.',
                        metavar='<filename>',
                        default='weights/model4.pth')
    parser.add_argument('--json',
                        help='Output labelme json.',
                        metavar='<filename>')
    parser.add_argument('--pped',
                        help='Output pre-processed image.',
                        metavar='<filename>')
    parser.add_argument('--npy',
                        help='Output heatmap as np.array. (e.g. heatmap.npy)',
                        metavar='<filename>')
    parser.add_argument('--heatmap',
                        help='Output heatmap image. (e.g. heatmap.jpg)',
                        metavar='<filename>')
    parser.add_argument('--affinity',
                        help='Output affinity image. (e.g. affinity.jpg)',
                        metavar='<filename>')
    parser.add_argument('--flip',
                        help='Horizontally flip input image.',
                        action='store_true')
    parser.add_argument('--height',
                        help='Image height.',
                        default=768,
                        type=int,
                        metavar='<height>')
    parser.add_argument('-v',
                        '--verbose',
                        help='Verbose mode.',
                        action='store_true')
    #args = parser.parse_args()
    #return args

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')

    args = parser.parse_args()
    return args


    print('Load', args.input, end=' ')
    original_img = load_image(args.input)
    #flip
    if args.flip:
        original_img = np.flip(original_img, axis=1)
    print('done.')
    original_shape = original_img.shape
    logger.debug('Original shape %s', original_shape)
    img = resize_height(args.height, original_img)
    resized_shape = img.shape
    if original_img.dtype != np.uint8:
        original_img = normalize_intensity(original_img)
    down_factor = 256
    new_shape = (down_factor *
                 np.ceil(np.array(img.shape) / down_factor)).astype(np.int64)
    img = pad_to_shape(img, new_shape)
    padded_shape = img.shape
    logger.debug('Input shape %s', padded_shape)

    img = apply_clahe(img)

    #preprocesed image
    if args.pped:
        print('Save pre-processed image: ', args.pped, end=' ')
        skimage.io.imsave(args.pped, img)
        print('done')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Load model', end=' ')
    model = create_model()
    model.load_state_dict(torch.load(args.weights))
    print('done.')
    model.to(device)
    model.eval()
    x = base_transform(img)
    x = torch.from_numpy(x[np.newaxis]).to(device)
    print('Calculate heatmaps', end=' ')
    with torch.no_grad():
        y = model(x).to('cpu').detach().numpy().copy()
        if args.flip:
            y = np.flip(y, axis=-1)
            original_img = np.flip(original_img, axis=1)
    print('done.')
    y = y.squeeze()

    #flip
    if args.flip:
        y = y[:, (y.shape[1] - resized_shape[0]):, (y.shape[2] - resized_shape[1]):]  # remove paddings
    else:
        y = y[:, :resized_shape[0], :resized_shape[1]]  # remove paddings
    y = np.clip(np.round(y.squeeze() * 255), 0, 255).astype(np.uint8)

    #npy
    if args.npy:
        print('Save raw heatmap: ', args.npy, end=' ')
        np.save(args.npy, y)
        print('done')
    #heatmap
    if args.heatmap:
        print('Save heatmap: ', args.heatmap, end=' ')
        rgb = heatmap2rgb(
            y[:4],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]],
        )
        rgb = skimage.transform.resize(rgb,
                                       original_shape,
                                       preserve_range=True)
        overlay = rgb_on_gray(rgb, original_img)
        skimage.io.imsave(args.heatmap, overlay, check_contrast=False)
        print('done')
    #affinity
    if args.affinity:
        print('Save affinity map: ', args.affinity, end=' ')
        rgb = heatmap2rgb(
            y[4:],
            [[1, 0, 0], [0, 1, 0]],
        )
        rgb = skimage.transform.resize(rgb,
                                       original_shape,
                                       preserve_range=True)
        overlay = rgb_on_gray(rgb, original_img)
        skimage.io.imsave(args.affinity, overlay, check_contrast=False)
        print('done')

    kps, scores = heatmap2points(y)
    kps = np.array(kps)
    logger.debug('Scores:%s', scores)
    kps = (original_shape[0] / resized_shape[0]) * kps
    logger.debug('Key points: %s', kps)

    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": Path(args.input).name,
        "imageData": None,
        "imageHeight": original_shape[0],
        "imageWidth": original_shape[1]
    }
    shapes = json_data['shapes']
    for kp, label in zip(kps, KP_LABELS):
        shapes.append({
            "label": label,
            "points": [kp.tolist()],
            "group_id": None,
            "shape_type": "point",
            "flags": {}
        })
    #json
    if args.json:
        print('Save json: ', args.json, end=' ')
        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(' done.')
    #output
    print('Save ', args.output, end=' ')
    pil = Image.fromarray(original_img).convert('RGB')
    cobb_draw.draw_cobb(pil, json_data)
    pil.save(args.output)
    print('done.')

    return 0

if __name__ == '__main__':
    sys.exit(parse_args())



""" if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        is_object = train.Network(args)
        is_object.train_network(args)
    elif args.phase == 'test':
        is_object = test.Network(args)
        is_object.test(args, save=False)
    elif args.phase == 'eval':
        is_object = eval.Network(args)
        is_object.eval(args, save=False)
        # is_object.eval_three_angles(args, save=False)  """

        