import torch
import numpy as np
from models import spinal_net
import cv2
import decoder
import os
from dataset import BaseDataset
import draw_points
import time
import cobb_evaluate

def apply_mask(image, mask, alpha=0.5):
    
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def eval(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']
            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder('imp2.jpg'))
            
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                   landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image))

class Network1(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def test(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio

            print('totol pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
                                                                                          ori_image_regress,
                                                                                          ori_image_points)

            cv2.imwrite('ori_image_regress_{}.jpg'.format(cnt), ori_image_regress)
            cv2.imwrite('ori_image_points_{}.jpg'.format(cnt), ori_image_points)

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()