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

import csv
import cv2

import datasets
from utils import flow_viz
from utils import frame_utils
from utils import error_viz

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--output_path', help="dataset for evaluation")
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
        else:
            raise ValueError(f"args.dataset=\'{args.dataset}\' not implemented")


