# Using VSCode debugger to peer into architecture
# an attempt at faster code reading
import sys
sys.path.append('core')

import torch
import argparse

from flowdiffuser import FlowDiffuser


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

    image1 = torch.randn([1, 3, 200, 400]).cuda()
    image2 = torch.randn([1, 3, 200, 400]).cuda()

    flow_something, flow_pr = model(image1=image1, image2=image2, iters=6, test_mode=True)

    flow_gt = torch.randn([1, 2, 200, 400]).cuda()
    valid = torch.ones([1, 1, 200, 400]).cuda()
    model.train()
    flow_predictions = model(image1=image1, image2=image2, iters=12, flow_gt=flow_gt, test_mode=False)

    #loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

    print('')