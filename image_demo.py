'''
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
'''




# 위는 원본 코드 아래는 수정 코드(23.08.29)
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import os
import time
import math


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', 
                        default=None, 
                        help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--draw-heatmap', action='store_true', 
                        default=True, help='Visualize the predicted heatmap')
    
    parser.add_argument('--show-kpt-idx', action='store_true', 
                        default=False, 
                        help='Whether to show the index of keypoints')
        # 이미지 별 예측 관절 키포인트 표시 유무

    parser.add_argument('--skeleton-style', default='mmpose', type=str, 
                        choices=['mmpose', 'openpose'], help='Skeleton style selection')
    
    parser.add_argument('--kpt-thr', type=float, 
                        default=0.1, help='Visualizing keypoint thresholds')
        # 키포인트 임계값 시각화

    parser.add_argument('--radius', type=int, 
                        default=1, help='Keypoint radius for visualization')
        # 3. 시각화를 위한 키포인트 반지름

    parser.add_argument('--thickness', type=int, 
                        default=2, 
                        help='Link thickness for visualization')
    
    parser.add_argument('--alpha', type=float, 
                        default=0.8, 
                        help='The transparency of bboxes')
        # 0.8이 기본. bbox의 투명성

    parser.add_argument('--show', action='store_true', 
                        default=True, 
                        help='whether to show img')
    
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    print("터미널 확인용 :",args)    
    # print(args.img.split('/')[-1])

    origin_file_path = args.img
    save_dir=  args.out_file

    for file in sorted(os.listdir(args.img)):
        '''
        파일 통째로 데이터셋으로 할 경우 
        '''
        start = time.time() #?

        # print(file)
        args.img = origin_file_path+file
        print("args.img :: ", args.img)
        print("터미널 확인 ", args.out_file+'/'+args.img.split('/')[-1])
        args.out_file = save_dir+args.img.split('/')[-1] # 경로를 새로 만들어서 지정해주기
        print("터미널 확인 args.out_file :",args.out_file)


        # build the model from a config file and a checkpoint file
        if args.draw_heatmap:
            cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        else:
            cfg_options = None

        # print("cfg_options : ", cfg_options)
            # -> {'model': {'test_cfg': {'output_heatmaps': True}}}

        model = init_model(
            args.config,
            args.checkpoint,
            device=args.device,
            cfg_options=cfg_options)

        # init visualizer
        model.cfg.visualizer.radius = args.radius
        model.cfg.visualizer.alpha = args.alpha
        model.cfg.visualizer.line_width = args.thickness

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.set_dataset_meta(
            model.dataset_meta, skeleton_style=args.skeleton_style)
        # print("visualizer:", visualizer)


        # inference a single image
        batch_results = inference_topdown(model, args.img)
            # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/apis/inference.py

        results = merge_data_samples(batch_results)
            # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/structures/utils.py
        
        # print(batch_results)


        end = time.time() # ?
        print(f'TIME : {end-start:.5f} sec') # ?


        # show the results
        img = imread(args.img, channel_order='rgb')
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            out_file=args.out_file)
        # esc 눌러야 정상종료됨

        # 아래는 초기화
        args.img=''
        args.out_file =''
        

if __name__ == '__main__':
    main()