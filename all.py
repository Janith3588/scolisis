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



#evel

def apply_mask(image, mask, alpha=0.5):

    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
    return image

class Network1(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,  # cen, tl, tr, bl, br
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
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            pr_cobb_angles.append(cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = np.sum(out_abs, axis=1)
        term2 = np.sum(out_add, axis=1)

        SMAPE = np.mean(term1 / term2 * 100)

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))
        print('SMAPE is {}'.format(SMAPE))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))


    def SMAPE_single_angle(self, gt_cobb_angles, pr_cobb_angles):
        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = out_abs
        term2 = out_add

        term2[term2==0] += 1e-5

        SMAPE = np.mean(term1 / term2 * 100)
        return SMAPE

    def eval_three_angles(self, args, save):
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
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            pr_cobb_angles.append(cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)


        print('SMAPE1 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,0], pr_cobb_angles[:,0])))
        print('SMAPE2 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,1], pr_cobb_angles[:,1])))
        print('SMAPE3 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,2], pr_cobb_angles[:,2])))

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))

##Test

""" def apply_mask(image, mask, alpha=0.5):
    
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image """

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
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio

            print('totol pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

            cobb_angles = cobb_angle_calc(pts0, ori_image_points)  ##add by me

            

            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
                                                                                          ori_image_regress,
                                                                                          ori_image_points)

            #cv2.imshow('ori_image_regress.jpg', ori_image_regress)   #commented by myself
            #cv2.imshow('ori_image_points.jpg', ori_image_points)     #commented by my self

            cv2.imwrite('ori_image_regress_{}.jpg'.format(cnt), ori_image_regress)
            cv2.imwrite('ori_image_points_{}.jpg'.format(cnt), ori_image_points)

            

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()

#Cobb

def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True

def cobb_angle_calc(pts, image):
    pts = np.asarray(pts, np.float32)   # 68 x 2
    h,w,c = image.shape
    num_pts = pts.shape[0]   # number of points, 68
    vnum = num_pts//4-1

    if pts.shape[0] % 2 != 0:
        pts = pts[:-1, :]  # remove the last row if number of rows is odd

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2

    for pt in mid_p:
        cv2.circle(image,
                   (int(pt[0]), int(pt[1])),
                   12, (0,255,255), -1, 1)

    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(image,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,0,255),
                 thickness=5, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    angles = np.arccos(cosine_angles)   # 17 x 17
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)
    cobb_angle1 = np.amax(maxt)
    cobb_angle1 = cobb_angle1/np.pi*180
    flag_s = is_S(mid_p_v)
    if not flag_s: # not S
        print('Not S')
        cobb_angle2 = angles[0, pos2]/np.pi*180
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        cv2.line(image,
                 (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
                 (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)
        cv2.line(image,
                 (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                 (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)

    else:
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            print('Is S: condition1')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            pos1_2 = pos1_2 + pos1[pos2]-1

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

        else:
            print('Is S: condition2')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1_1, :(pos1_1+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

    print("cob1: "+ str(cobb_angle1),"cobb2 : " + str(cobb_angle2),"Cobb3 : " +  str(cobb_angle3))
    return [cobb_angle1, cobb_angle2, cobb_angle3]