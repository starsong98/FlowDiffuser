import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy

import csv
import cv2

import datasets
from utils import flow_viz
from utils import frame_utils
from utils import error_viz
from utils import warp_utils

from flowdiffuser import FlowDiffuser

from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            # if sequence != sequence_prev:
            #     flow_prev = None
            
            if (sequence != sequence_prev) or (dstype == 'final' and sequence in ['market_4', ]) or dstype == 'clean':
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)   # flow is numpy ndarray, shape (H, W, 2)


@torch.no_grad()
def validate_chairs(model, iters=24, output_path=None, split='validation'):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    # detailed stat saving - part 1 - header
    if output_path is not None:
        lines_to_save = [['filename0', 'filename1', 'epe']]
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    if split == 'validation':
        val_dataset = datasets.FlyingChairs(split='validation')
        out_filename = "FlyingChairs-val-stats.csv"
        if output_path is not None:
            out_vis_dir = os.path.join(output_path, 'chairs-validation')
        print("Validation on FlyingChairs validation split")
    elif split == 'training':
        val_dataset = datasets.FlyingChairs(split='training')
        out_filename = "FlyingChairs-train-stats.csv"
        if output_path is not None:
            out_vis_dir = os.path.join(output_path, 'chairs-training')
        print("Validation on FlyingChairs training split")
    elif split == 'fcdn-val':
        val_dataset = datasets.FlyingChairs(split='validation', root='datasets/FCDN/data')
        out_filename = "FCDN-val-stats.csv"
        if output_path is not None:
            out_vis_dir = os.path.join(output_path, 'fcdn-validation')
        print("Validation on FCDN validation split")
    elif split == 'fcdn-train':
        val_dataset = datasets.FlyingChairs(split='training', root='datasets/FCDN/data')
        out_filename = "FCDN-train-stats.csv"
        if output_path is not None:
            out_vis_dir = os.path.join(output_path, 'fcdn-training')
        print("Validation on FCDN training split")
    else:
        raise ValueError("split type not implemented")

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        img_pair_overlay = ((image1 + image2) / 2).permute(1, 2, 0)    # for later
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        err = epe.clone()   # for later
        epe_list.append(epe.view(-1).numpy())

        # detailed stat saving - part 2 & 3 - individual sample handling
        if output_path is not None:
            # detailed stat saving - part 2 - individual stats
            filename0 = val_dataset.image_list[val_id][0]
            filename1 = val_dataset.image_list[val_id][1]
            epe_single = epe.mean().cpu().item()
            lines_to_save.append([filename0, filename1, epe_single])

            # detailed stat saving - part 3 - visuals
            #out_vis_dir = os.path.join(output_path, 'chairs-validation')
            #out_vis_dir = os.path.join(output_path, 'chairs-training')
            if not os.path.isdir(out_vis_dir):
                os.makedirs(out_vis_dir)
            out_vis_path = os.path.join(out_vis_dir, os.path.basename(filename0).replace('.ppm', '.png'))
            #gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt[0].permute(1, 2, 0).cpu().numpy())
            gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy())
            pred_vis = flow_viz.flow_to_image(flow_uv=flow_pr[0].permute(1, 2, 0).cpu().numpy())
            epe_vis = error_viz.visualize_error_map(err.cpu().numpy())
            combined_vis = np.concatenate([img_pair_overlay, pred_vis, gt_vis, epe_vis], axis=0)
            combined_vis = np.flip(combined_vis, axis=2)
            cv2.imwrite(out_vis_path, combined_vis)

            # detailed stat saving - part 3.5 - flow files
            flowname = val_dataset.flow_list[val_id]
            out_flow_path = os.path.join(out_vis_dir, os.path.basename(flowname))
            #print(f'supposed to write to {out_flow_path}')
            frame_utils.writeFlow(out_flow_path, flow_pr[0].permute(1, 2, 0).cpu().numpy())
            #break

    epe = np.mean(np.concatenate(epe_list))

    # detailed stat saving - part 4 - average stats
    if output_path is not None:
        lines_to_save.append(['Averaged_stats', '', epe])
        #out_filename = "FlyingChairs-val-stats.csv"
        #out_filename = "FlyingChairs-train-stats.csv"
        stat_path = os.path.join(output_path, out_filename)
        with open(stat_path, 'a+', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerows(lines_to_save)

    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_autoflow(model, iters=24, output_path=None, split='subval'):
    """ Perform evaluation on the AutoFlow datset """
    model.eval()
    epe_list = []
    # detailed stat saving - part 1 - header
    if output_path is not None:
        lines_to_save = [['filename0', 'filename1', 'epe']]
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    if split == 'train':
        val_dataset = datasets.AutoFlow(split='train')
        out_filename = "AutoFlow-full-stats.csv"
        out_vis_dir = os.path.join(output_path, 'autoflow-full')
        print("Validation on full AutoFlow dataset")
    elif split == 'subtrain':
        val_dataset = datasets.AutoFlow(split='subtrain')
        out_filename = "AutoFlow-subtrain-stats.csv"
        out_vis_dir = os.path.join(output_path, 'autoflow-subtrain')
        print("Validation on AutoFlow training split")
    elif split == 'subval':
        val_dataset = datasets.AutoFlow(split='subval')
        out_filename = "AutoFlow-subval-stats.csv"
        out_vis_dir = os.path.join(output_path, 'autoflow-subval')
        print("Validation on AutoFlow validation split")
    else:
        raise ValueError("split type not implemented")

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        img_pair_overlay = ((image1 + image2) / 2).permute(1, 2, 0)    # for later
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        err = epe.clone()   # for later
        epe_list.append(epe.view(-1).numpy())

        # detailed stat saving - part 2 & 3 - individual sample handling
        if output_path is not None:
            # detailed stat saving - part 2 - individual stats
            filename0 = val_dataset.image_list[val_id][0]
            filename1 = val_dataset.image_list[val_id][1]
            epe_single = epe.mean().cpu().item()
            lines_to_save.append([filename0, filename1, epe_single])

            # detailed stat saving - part 3 - visuals
            #out_vis_dir = os.path.join(output_path, 'chairs-validation')
            #out_vis_dir = os.path.join(output_path, 'chairs-training')
            if not os.path.isdir(out_vis_dir):
                os.makedirs(out_vis_dir)
            out_vis_path = os.path.join(out_vis_dir, os.path.basename(filename0).replace('.ppm', '.png'))
            #gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt[0].permute(1, 2, 0).cpu().numpy())
            gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy())
            pred_vis = flow_viz.flow_to_image(flow_uv=flow_pr[0].permute(1, 2, 0).cpu().numpy())
            epe_vis = error_viz.visualize_error_map(err.cpu().numpy())
            combined_vis = np.concatenate([img_pair_overlay, pred_vis, gt_vis, epe_vis], axis=0)
            combined_vis = np.flip(combined_vis, axis=2)
            cv2.imwrite(out_vis_path, combined_vis)

            # detailed stat saving - part 3.5 - flow files
            flowname = val_dataset.flow_list[val_id]
            out_flow_path = os.path.join(out_vis_dir, os.path.basename(flowname))
            #print(f'supposed to write to {out_flow_path}')
            frame_utils.writeFlow(out_flow_path, flow_pr[0].permute(1, 2, 0).cpu().numpy())
            #break

    epe = np.mean(np.concatenate(epe_list))

    # detailed stat saving - part 4 - average stats
    if output_path is not None:
        lines_to_save.append(['Averaged_stats', '', epe])
        #out_filename = "FlyingChairs-val-stats.csv"
        #out_filename = "FlyingChairs-train-stats.csv"
        stat_path = os.path.join(output_path, out_filename)
        with open(stat_path, 'a+', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerows(lines_to_save)

    print("Validation AutoFlow EPE: %f" % epe)
    return {'autoflow': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, output_path=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        # detailed stat saving - part 1 - header
        if output_path is not None:
            lines_to_save = [['filename0', 'filename1', 'epe']]
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

        for val_id in tqdm(range(len(val_dataset)), desc=f"Validation on Sintel-train-{dstype}:"):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            img_pair_overlay = ((image1 + image2) / 2).permute(1, 2, 0)    # for later
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            err = epe.clone()   # for later
            epe_list.append(epe.view(-1).numpy())

            # detailed stat saving - part 2 & 3 - individual sample handling
            if output_path is not None:
                # detailed stat saving - part 2 - individual stats
                filename0 = val_dataset.image_list[val_id][0]
                filename1 = val_dataset.image_list[val_id][1]
                epe_single = epe.mean().cpu().item()
                lines_to_save.append([filename0, filename1, epe_single])

                # detailed stat saving - part 3 - visuals
                #out_vis_dir = os.path.join(output_path, f'Sintel-train-{dstype}')
                filenames_split = filename0.split('/')
                out_vis_dir = os.path.join(output_path, 'Sintel-train', dstype, filenames_split[-2])
                if not os.path.isdir(out_vis_dir):
                    os.makedirs(out_vis_dir)
                out_vis_path = os.path.join(out_vis_dir, os.path.basename(filename0))
                #gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt[0].permute(1, 2, 0).cpu().numpy())
                gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy())
                pred_vis = flow_viz.flow_to_image(flow_uv=flow_pr[0].permute(1, 2, 0).cpu().numpy())
                epe_vis = error_viz.visualize_error_map(err.cpu().numpy())
                combined_vis = np.concatenate([img_pair_overlay, pred_vis, gt_vis, epe_vis], axis=0)
                combined_vis = np.flip(combined_vis, axis=2)
                cv2.imwrite(out_vis_path, combined_vis)

                # detailed stat saving - part 3.5 - flow files
                flowname = val_dataset.flow_list[val_id]
                #out_flow_dir = os.path.join(out_vis_dir, 'raw_flows')
                #if not os.path.isdir(out_flow_dir):
                #    os.makedirs(out_flow_dir)
                #out_flow_path = os.path.join(out_flow_dir, os.path.basename(flowname))
                out_flow_path = os.path.join(out_vis_dir, os.path.basename(flowname))
                #print(f'supposed to write to {out_flow_path}')
                frame_utils.writeFlow(out_flow_path, flow.permute(1, 2, 0).numpy())
                #break

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        # detailed stat saving - part 4 - average stats
        if output_path is not None:
            lines_to_save.append(['Averaged_stats', '', epe])
            #out_filename = "FlyingChairs-val-stats.csv"
            out_filename = f"Sintel-train-{dstype}.csv"
            stat_path = os.path.join(output_path, out_filename)
            with open(stat_path, 'a+', newline="") as fp:
                writer = csv.writer(fp)
                writer.writerows(lines_to_save)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24, output_path=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    # detailed stat saving - part 1 - header
    if output_path is not None:
        lines_to_save = [['filename0', 'filename1', 'kitti-epe', 'kitti-f1']]
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    for val_id in tqdm(range(len(val_dataset)), desc="Validation on KITTI-15-train:"):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        img_pair_overlay = ((image1 + image2) / 2).permute(1, 2, 0)    # for later
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()
        err = epe.clone()   # for later

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # detailed stat saving - part 2 & 3 - individual sample handling
        if output_path is not None:
            # detailed stat saving - part 2 - individual stats
            filename0 = val_dataset.image_list[val_id][0]
            filename1 = val_dataset.image_list[val_id][1]
            epe_single = epe[val].mean().cpu().item()
            f1_single = 100 * out[val].mean().item()
            lines_to_save.append([filename0, filename1, epe_single, f1_single])

            # detailed stat saving - part 3 - visuals
            out_vis_dir = os.path.join(output_path, 'KITTI15-train')
            if not os.path.isdir(out_vis_dir):
                os.makedirs(out_vis_dir)
            out_vis_path = os.path.join(out_vis_dir, os.path.basename(filename0))
            #gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt[0].permute(1, 2, 0).cpu().numpy())
            gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy())
            pred_vis = flow_viz.flow_to_image(flow_uv=flow.permute(1, 2, 0).cpu().numpy())
            valid_mask = (valid_gt >= 0.5).cpu()
            epe_vis = error_viz.visualize_error_map(err.cpu().numpy(), valid_mask)
            combined_vis = np.concatenate([img_pair_overlay, pred_vis, gt_vis, epe_vis], axis=0)
            combined_vis = np.flip(combined_vis, axis=2)
            cv2.imwrite(out_vis_path, combined_vis)

            # detailed stat saving - part 3.5 - flow files
            flowname = val_dataset.flow_list[val_id]
            out_flow_dir = os.path.join(out_vis_dir, 'raw_flows')
            if not os.path.isdir(out_flow_dir):
                os.makedirs(out_flow_dir)
            out_flow_path = os.path.join(out_flow_dir, os.path.basename(flowname))
            #print(f'supposed to write to {out_flow_path}')
            frame_utils.writeFlowKITTI(out_flow_path, flow.permute(1, 2, 0).numpy())
            #break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    # detailed stat saving - part 4 - average stats
    if output_path is not None:
        lines_to_save.append(['Averaged_stats', '', epe, f1])
        out_filename = "KITTI15-train-stats.csv"
        stat_path = os.path.join(output_path, out_filename)
        with open(stat_path, 'a+', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerows(lines_to_save)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_things(model, iters=32, output_path=None):
    """ Peform validation using the FlyingThings3d (test) split """
    model.eval()
    results = {}
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3DTest(dstype=dstype)    # C+T was indeed trained using both rendering psases
        epe_list = []
        # detailed stat saving - part 1 - header
        if output_path is not None:
            lines_to_save = [['filename0', 'filename1', 'flowname', 'epe']]
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

        len_val = len(val_dataset) if output_path is not None else 800
        #for val_id in tqdm(range(len(val_dataset)), desc=f"Validation on FT3D-TEST-{dstype}:"):
        for val_id in tqdm(range(len_val), desc=f"Validation on FT3D-TEST-{dstype}:"):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            img_pair_overlay = ((image1 + image2) / 2).permute(1, 2, 0)    # for later
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()  # [H, W]
            err = epe.clone()   # for later; [H, W]
            #epe_list.append(epe.view(-1).numpy())
            val = valid_gt >= 0.5   # [H, W]
            #epe_list.append(epe[val].view(-1).numpy())
            epe_list.append(epe[val].mean().item())

            # detailed stat saving - part 2 & 3 - individual sample handling
            if output_path is not None:
                # detailed stat saving - part 2 - individual stats
                filename0 = val_dataset.image_list[val_id][0]
                filename1 = val_dataset.image_list[val_id][1]
                flowname = val_dataset.flow_list[val_id]
                epe_single = epe.mean().cpu().item()
                
                out_vis_dir = os.path.join(output_path, f'Things-test/{dstype}')
                #sub1, sub2, sub3, visname = flowname.split('/')[-4:]
                sub_A, sub_seq, sub_t, sub_lr, visname = flowname.split('/')[-5:]
                #sub01, sub02, sub03, name0 = filename0.split('/')[-4:]
                #sub11, sub12, sub13, name1 = filename1.split('/')[-4:]
                #out_flow_path = os.path.join(out_vis_dir, sub1, sub2, sub3, visname)
                out_flow_path = os.path.join(out_vis_dir, f"{sub_A}-{sub_seq}", f"{sub_t}-{visname[-10:]}")
                out_vis_path = out_flow_path.replace('.pfm', '.png')
                if not os.path.isdir(os.path.dirname(out_flow_path)):
                    os.makedirs(os.path.dirname(out_flow_path))
                lines_to_save.append([filename0, filename1, flowname, epe_single])
                #lines_to_save.append([
                #    os.path.join(sub01, sub02, sub03, name0),   # filename0, trimmed
                #    os.path.join(sub11, sub12, sub13, name1),   # filename1, trimmed
                #    os.path.join(sub1, sub2, sub3, visname),   # flowname, trimmed
                #    epe_single])
                #break

                # detailed stat saving - part 3 - visuals
                #gt_vis = flow_viz.flow_to_image(flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy())
                #pred_vis = flow_viz.flow_to_image(flow_uv=flow_pr[0].permute(1, 2, 0).cpu().numpy())           
                pred_vis, gt_vis = error_viz.compare_flow_viz(
                    out_flow_uv=flow_pr[0].permute(1, 2, 0).cpu().numpy(),
                    gt_flow_uv=flow_gt.permute(1, 2, 0).cpu().numpy(),
                    valid_mask=val,
                )
                epe_vis = error_viz.visualize_error_map(err.cpu().numpy())
                combined_vis = np.concatenate([img_pair_overlay, pred_vis, gt_vis, epe_vis], axis=0)
                #combined_vis = np.concatenate([img_pair_overlay, epe_vis], axis=0)
                #ombined_vis = np.flip(combined_vis, axis=2)
                cv2.imwrite(out_vis_path, combined_vis)

                # detailed stat saving - part 3.5 - flow files
                frame_utils.writeFlow(out_flow_path, flow.permute(1, 2, 0).numpy())
                #break

        #epe_all = np.concatenate(epe_list)
        epe_all = np.array(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        # detailed stat saving - part 4 - average stats
        if output_path is not None:
            lines_to_save.append(['Averaged_stats', '', '', epe])
            #out_filename = "FlyingChairs-val-stats.csv"
            out_filename = f"FlyingThings3D-test-{dstype}.csv"
            stat_path = os.path.join(output_path, out_filename)
            with open(stat_path, 'a+', newline="") as fp:
                writer = csv.writer(fp)
                writer.writerows(lines_to_save)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


def infer_singleclip(model, output_path=None, iters=32, scale_factor=0., clip_root=None):
    assert clip_root is not None
    test_dataset = datasets.RealVideoSingle(root=clip_root)

    # save directory prep.
    os.makedirs(output_path, exist_ok=True)
    _, _, (sequence, _) = test_dataset[0]
    flowdir_path = os.path.join(output_path, sequence, 'flow')
    flowvis_path = os.path.join(output_path, sequence, 'flow_vis')
    fwarp_path = os.path.join(output_path, sequence, 'image1_fwarp')
    bwarp_path = os.path.join(output_path, sequence, 'image2_bwarp')
    grid_path = os.path.join(output_path, sequence, 'tile4_vis')
    os.makedirs(flowdir_path, exist_ok=True)
    os.makedirs(flowvis_path, exist_ok=True)
    os.makedirs(fwarp_path, exist_ok=True)
    os.makedirs(bwarp_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)

    model.eval()
    for test_id in tqdm(range(len(test_dataset)), desc='processing sequence...'):
        image1, image2, (sequence, frame_id) = test_dataset[test_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()

        # downsample-upsample; see SEA-RAFT
        #if scale_factor != 0.0:
        # downsample input
        img1 = F.interpolate(image1, scale_factor=2 ** scale_factor, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** scale_factor, mode='bilinear', align_corners=False)
        
        # padding needed
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        # forward pass
        #flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
        flow_low, flow_pr = model(img1, img2, iters=iters, flow_init=None, test_mode=True)

        # unpad flow
        flow_pr = padder.unpad(flow_pr)

        # upsample output
        #flow_down = F.interpolate(flow_pr, scale_factor=0.5 ** scale_factor, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        flow = F.interpolate(flow_pr, scale_factor=0.5 ** scale_factor, mode='bilinear', align_corners=False) * (0.5 ** args.scale_factor)
        flo = flow.clone()
        flow = flow[0].permute(1, 2, 0).cpu()

        # save flow file - sintel flo file
        output_filename = os.path.join(flowdir_path, f'frame_{(frame_id+1):04d}.flo')
        frame_utils.writeFlow(output_filename, flow)

        # save flow visualization
        flow_vis = flow_viz.flow_to_image(flow.numpy())
        #output_visname = os.path.join(output_path, 'flow_vis', f'frame_{(frame_id+1):04d}.png')
        output_visname = os.path.join(flowvis_path, f'frame_{(frame_id+1):04d}.png')
        cv2.imwrite(output_visname, flow_vis[:, :, [2,1,0]])

        # forward warp and save
        image1_fwp, _ = warp_utils.fwarp_wrapper(img=image1, flo=flo)
        #output_fwarpname = os.path.join(output_path, 'image1_fwarp', f'frame_{(frame_id+1):04d}.png')
        output_fwarpname = os.path.join(fwarp_path, f'frame_{(frame_id+1):04d}.png')
        cv2.imwrite(output_fwarpname, image1_fwp[:, :, [2,1,0]])

        # grid visualization
        image1 = image1[0].permute(1, 2, 0).cpu().numpy()
        image2 = image2[0].permute(1, 2, 0).cpu().numpy()
        row_1 = np.concatenate([image1, image2], axis=1)
        row_2 = np.concatenate([flow_vis, image1_fwp], axis=1)
        grid_vis = np.concatenate([row_1, row_2], axis=0)
        grid_vis_path = os.path.join(grid_path, f'frames_{(frame_id+1):04d}_{(frame_id+2):04d}.png')
        cv2.imwrite(grid_vis_path, grid_vis[:, :, [2,1,0]])
        
        print()


def infer_singleclip_ensemble(model, output_path=None, iters=32, scale_factor=0., clip_root=None, ensembles=3):
    assert clip_root is not None
    test_dataset = datasets.RealVideoSingle(root=clip_root)

    # save directory prep.
    os.makedirs(output_path, exist_ok=True)
    _, _, (sequence, _) = test_dataset[0]
    flowdir_path = os.path.join(output_path, sequence, 'flow')
    flowvis_path = os.path.join(output_path, sequence, 'flow_vis')
    fwarp_path = os.path.join(output_path, sequence, 'image1_fwarp')
    bwarp_path = os.path.join(output_path, sequence, 'image2_bwarp')
    grid_path = os.path.join(output_path, sequence, 'tile4_vis')
    var_path = os.path.join(output_path, sequence, 'variance')
    heatmap_path = os.path.join(output_path, sequence, 'heatmap_vis')
    summary_path = os.path.join(output_path, sequence, 'tile_ensemble_vis')
    os.makedirs(flowdir_path, exist_ok=True)
    os.makedirs(flowvis_path, exist_ok=True)
    os.makedirs(fwarp_path, exist_ok=True)
    os.makedirs(bwarp_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(var_path, exist_ok=True)
    os.makedirs(heatmap_path, exist_ok=True)
    
    # prep different directories for each run
    rundirs = []
    for i in range(ensembles):
        rundirs.append(f'run_{i}')
    rundirs.append(f'average')
    list_temp = [flowdir_path, flowvis_path, fwarp_path, bwarp_path, grid_path]
    for dir_temp in list_temp:
        for rundir in rundirs:
            os.makedirs(os.path.join(dir_temp, rundir), exist_ok=True)

    model.eval()
    for test_id in tqdm(range(len(test_dataset)), desc='processing sequence...'):
        image1, image2, (sequence, frame_id) = test_dataset[test_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()

        scale_factor_vis = 0.125

        # downsample-upsample; see SEA-RAFT
        #if scale_factor != 0.0:
        # downsample input
        img1 = F.interpolate(image1, scale_factor=2 ** scale_factor, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** scale_factor, mode='bilinear', align_corners=False)
        
        # padding needed
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        # forward pass, ensembled
        flows_pr = []
        for i in tqdm(range(ensembles), desc=f'ensemble run on frame pair #{frame_id}'):
            seed = i
            torch.manual_seed(seed)
            flow_low, flow_pr = model(img1, img2, iters=iters, flow_init=None, test_mode=True)
            flows_pr.append(flow_pr)
        flow_pr = torch.concatenate(flows_pr, dim=0)    # [N, 2, H', W']
        #print()

        # unpad flow
        flow_pr = padder.unpad(flow_pr)

        # upsample output
        #flow_down = F.interpolate(flow_pr, scale_factor=0.5 ** scale_factor, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
        flow = F.interpolate(flow_pr, scale_factor=0.5 ** scale_factor, mode='bilinear', align_corners=False) * (0.5 ** args.scale_factor)

        # compute average flow and variance
        mean_flow = torch.mean(flow, dim=0) # [2, H, W]
        variance = torch.mean((flow - mean_flow)**2, dim=0) # [2, H, W]
        var_mag = torch.sqrt(torch.sum(variance**2, dim=0)) # [H, W]

        # attach average flow
        flow = torch.cat([flow, mean_flow[None]], dim=0)    # [N+1, 2, H, W]
        
        # try saving the variance & heatmap
        var_filename = os.path.join(var_path, f'frame_{(frame_id+1):04d}.npy')
        np.save(var_filename, variance.cpu().numpy())
        heatmap_filename = os.path.join(heatmap_path, f'frame_{(frame_id+1):04d}.png')
        var_mag = var_mag.cpu().numpy()
        fig, ax = plt.subplots(figsize=(var_mag.shape[1] / 100, var_mag.shape[0] / 100), dpi=100)
        cax = ax.imshow(var_mag, cmap='hot', interpolation='nearest', vmax=10)
        ax.axis('off')
        plt.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        var_mag_down = scipy.ndimage.zoom(var_mag, (scale_factor_vis, scale_factor_vis), order=1)

        flo = flow.clone()  # [N+1, 2, H, W]
        #flow = flow[0].permute(1, 2, 0).cpu()
        #image1_fwps, _ = warp_utils.fwarp_wrapper_ensemble(img=image1, flo=flo)
        #print()
        image1_np = image1[0].permute(1, 2, 0).cpu().numpy()
        image2_np = image2[0].permute(1, 2, 0).cpu().numpy()
        #scale_factor_vis = 0.25
        #image1_down = F.interpolate(image1, scale_factor=scale_factor_vis, mode='bilinear', align_corners=False)
        #image2_down = F.interpolate(image1, scale_factor=scale_factor_vis, mode='bilinear', align_corners=False)
        #image1_down = image1_down[0].permute(1, 2, 0).cpu().numpy()
        #image2_down = image2_down[0].permute(1, 2, 0).cpu().numpy()
        images_overlay = (image1 + image2) / 2.
        images_overlay_down = F.interpolate(images_overlay, scale_factor=scale_factor_vis, mode='bilinear', align_corners=False)
        images_overlay_down = images_overlay_down[0].permute(1, 2, 0).cpu().numpy()

        flow_vis_list = []
        image1_fwarps_list = []
        # now saving - loop needed
        for i in tqdm(range(flow.shape[0]), desc=f'ensemble postproc on frame pair #{frame_id}'):
            flow_single = flow[i].permute(1, 2, 0).cpu()
            #print() # dummy for pause points

            # save flow files - sintel flo file
            #output_filename = os.path.join(flowdir_path, f'frame_{(frame_id+1):04d}.flo')
            output_filename = os.path.join(flowdir_path, rundirs[i], f'frame_{(frame_id+1):04d}.flo')
            frame_utils.writeFlow(output_filename, flow_single)

            # save flow visualization
            flow_vis = flow_viz.flow_to_image(flow_single.numpy())
            #output_visname = os.path.join(output_path, 'flow_vis', f'frame_{(frame_id+1):04d}.png')
            #output_visname = os.path.join(flowvis_path, f'frame_{(frame_id+1):04d}.png')
            output_visname = os.path.join(flowvis_path, rundirs[i], f'frame_{(frame_id+1):04d}.png')
            cv2.imwrite(output_visname, flow_vis[:, :, [2,1,0]])
            flow_vis_list.append(scipy.ndimage.zoom(flow_vis, (scale_factor_vis, scale_factor_vis, 1), order=1))

            # forward warp and save
            #image1_fwp, _ = warp_utils.fwarp_wrapper(img=image1, flo=flo)
            image1_fwp, _ = warp_utils.fwarp_wrapper(img=image1, flo=flo[i][None])
            #output_fwarpname = os.path.join(output_path, 'image1_fwarp', f'frame_{(frame_id+1):04d}.png')
            output_fwarpname = os.path.join(fwarp_path, rundirs[i], f'frame_{(frame_id+1):04d}.png')
            cv2.imwrite(output_fwarpname, image1_fwp[:, :, [2,1,0]])
            image1_fwarps_list.append(scipy.ndimage.zoom(image1_fwp, (scale_factor_vis, scale_factor_vis, 1), order=1))
            #print()

            # individual run grid visualization - 2x bilinear downsampling via scipy
            row_1 = np.concatenate([image1_np, image2_np], axis=1)
            row_2 = np.concatenate([flow_vis, image1_fwp], axis=1)
            grid_vis = np.concatenate([row_1, row_2], axis=0)
            factor = 0.5
            grid_vis = scipy.ndimage.zoom(grid_vis, (factor, factor, 1), order=1)
            grid_vis_path = os.path.join(grid_path, rundirs[i], f'frames_{(frame_id+1):04d}_{(frame_id+2):04d}.png')
            cv2.imwrite(grid_vis_path, grid_vis[:, :, [2,1,0]])

        #print()
        # grid visualization - these should be scaled down though
        #image1 = image1[0].permute(1, 2, 0).cpu().numpy()
        #image2 = image2[0].permute(1, 2, 0).cpu().numpy()
        #row_1 = np.concatenate([image1, image2], axis=1)
        #row_2 = np.concatenate([flow_vis, image1_fwp], axis=1)
        #grid_vis = np.concatenate([row_1, row_2], axis=0)
        #grid_vis_path = os.path.join(grid_path, f'frames_{(frame_id+1):04d}_{(frame_id+2):04d}.png')
        #cv2.imwrite(grid_vis_path, grid_vis[:, :, [2,1,0]])

        # TODO plot ensemble summarization
        # compute figure dimensions
        n_cols = ensembles + 3
        H, W = images_overlay_down.shape[:2]

        # create the actual figure
        fig, axes = plt.subplots(2, n_cols, figsize=(W / 100 * n_cols, H / 100* 2 + 1), dpi=100)
        
        # plot overlay
        axes[0, 0].imshow(images_overlay_down / 255.)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Overlayed Inputs', fontsize=20)

        
        # plot heatmap
        axes[0, 1].imshow(var_mag_down, cmap='hot', interpolation='nearest', vmax=10)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Variance heat map', fontsize=20)
        #axes[0, 1].cbar()
        #fig.colorbar(cax, )

        # plot flow maps
        for i, flow_vis in enumerate(flow_vis_list):
            axes[0, i+2].imshow(flow_vis)
            axes[0, i+2].axis('off')
            axes[0, i+2].set_title(rundirs[i], fontsize=20)
        
        # plot forwarped frames
        for i, fwarped in enumerate(image1_fwarps_list):
            axes[1, i+2].imshow(fwarped / 255.)
            axes[1, i+2].axis('off')

        # remove redundant subplots
        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        summary_filename = os.path.join(summary_path, f'frames_{(frame_id+1):04d}_{(frame_id+2):04d}.png')
        plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0)
        plt.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--output_path', help="dataset for evaluation")
    parser.add_argument('--scale_factor', default=0.0, type=float, help="downsampling factor for large inputs. Input image is UPsampled by factor of 2^(scale_factor). Default is 0, i.e., no downsampling. Only applied for real video inference.")
    parser.add_argument('--clip_root', help="video frame sequence folder for inference")
    parser.add_argument('--ensembles', type=int, help="Number of different noise initializations for randomness")
    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, output_path=args.output_path)
        elif args.dataset == 'chairs-train':
            validate_chairs(model.module, output_path=args.output_path, split='training')
        elif args.dataset == 'fcdn-val':
            validate_chairs(model.module, output_path=args.output_path, split='fcdn-val')
        elif args.dataset == 'fcdn-train':
            validate_chairs(model.module, output_path=args.output_path, split='fcdn-train')

        elif args.dataset == 'autoflow':
            validate_autoflow(model.module, output_path=args.output_path, split='train')
        elif args.dataset == 'autoflow-subtrain':
            validate_autoflow(model.module, output_path=args.output_path, split='subtrain')
        elif args.dataset == 'autoflow-subval':
            validate_autoflow(model.module, output_path=args.output_path, split='subval')

        elif args.dataset == 'things':
            validate_things(model.module, output_path=args.output_path)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, output_path=args.output_path)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, output_path=args.output_path)
        
        elif args.dataset == 'kitti-submission':
            create_kitti_submission(model.module, output_path=args.output_path)
        elif args.dataset == 'sintel-submission':
            create_sintel_submission(model.module, output_path=args.output_path)

        elif args.dataset == 'infer_singleclip':
            infer_singleclip(
                model.module, output_path=args.output_path,
                scale_factor=args.scale_factor, clip_root=args.clip_root
            )
        elif args.dataset == 'infer_singleclip_ensemble':
            infer_singleclip_ensemble(
                model.module, output_path=args.output_path,
                scale_factor=args.scale_factor, clip_root=args.clip_root, ensembles=args.ensembles
            )

        else:
            raise ValueError(f"args.dataset=\'{args.dataset}\' not implemented")


