# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time, math
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """
    Visualize predicted keypoints (and heatmaps) of one image.
    이미지에 대한 예측 키포인트 (와 히트맵) 시각화.
    """

    # predict bbox (bbox 예측)
    start= time.time()
    det_result = inference_detector(detector, img) 
        # hrnettest/mmpose/mmpose/apis/inference.py 의 함수
    pred_instance = det_result.pred_instances.cpu().numpy()
        # .cpu() : GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사하는 method
        # .numpy() : tensor를 numpy로 변환하여 반환한다
        # gpu 메모리에 올려져있는 tensor를 numpy로 변환하기 위해서는 우선 cpu 메모리로 옮겨야한다
        # https://byeongjo-kim.tistory.com/32
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        # 축을 기준으로 배열을 합친다
        # 1차원에서는 axis=0은 행방향이나 열방향의 개념이 없음
        # 2차원에서는 axis=0은 행(위->아래)방향을 의미. axis=1은 열(좌->우)를 의미한다.(0,1 만 존재)
        # 3차원에서는 0은 높이 방향, 1은 행 방향, 2는 열방향 
        # https://pybasall.tistory.com/33
        # [:, None]는 행을 그대로 유지하면서, 열을 하나 더 추가하는 방법
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
        # https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
        # numpy.logical_and : numpy에서 제공하는 논리 연산 함수 중 하나
        # 각 element가 모든 조건을 충족할 경우 true, 하나라도 충족하지 못할 경우 false를 반환한다
        # args.det_cat_id, args.bbox_thr ?????????????

    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
        # hrnettest/mmpose/mmpose/evaluation/functional/nms.py
        #

    end = time.time()
    print(f'inference_detector TIME : {end-start:.5f} sec') 

    # predict keypoints (keypoint 예측)
    start= time.time()
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    end = time.time()
    print(f'pose_results TIME : {end-start:.5f} sec') 


    # print("확인 :",pose_results)
        # hrnettest/mmpose/mmpose/apis/inference.py 
    data_samples = merge_data_samples(pose_results)
        # hrnettest/mmpose/mmpose/structures/utils.py
    

    # show the results (결과 보여주기)
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
            # https://mmcv-jm.readthedocs.io/en/latest/image.html
            # 불러올 이미지와 채널(rgb or bgr)
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    # print("터미널 확인 type :::::", data_samples.get('pred_instances', None))
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', 
                        # default='/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
                        help='Config file for detection') # is not None 
    # mmet config file

    parser.add_argument('det_checkpoint',
                        # default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                        help='Checkpoint file for detection') # is not None
    # mmdet checkpoint file

    parser.add_argument('pose_config', 
                        # default='/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
                        help='Config file for pose')
    # mmpose config file

    parser.add_argument('pose_checkpoint', 
                        # default='/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
                        help='Checkpoint file for pose')
    # mmpose checkpoint file

    parser.add_argument('--input', type=str, 
                        # default='/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4', 
                        help='Image/Video file')
    # 입력 이미지
    
    parser.add_argument('--show', action='store_true',
                        default=False,
                        help='whether to show img')
    # 이미지를 보여줄지 말지
    
    parser.add_argument('--output-root',type=str,
                        # default='/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/demo/vis_results',
                        help='root of the output img file. ''Default not saving the visualization images.')
    # output 저장 경로, default는 저장하지 않음 
    
    parser.add_argument('--save-predictions',action='store_true',
                        default=True,
                        help='whether to save predicted results')
    # 예측 결과를 저장할것인가 여부

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # cpu 인지 gpu 인지 설정. default는 gpu
    
    parser.add_argument('--det-cat-id',type=int,default=0,help='Category id for bounding box detection model')
    # bbox detection model을 위한 카테고리 id?

    parser.add_argument('--bbox-thr',type=float,default=0.9,help='Bounding box score threshold')
    # bbox 임계값
    
    parser.add_argument('--nms-thr',type=float,default=0.4,help='IoU threshold for bounding box NMS')
    # bounding box NMS을 위한 iou 임계값
    # 숫자가 높아질 경우, 기존 0.3에서 탐지하지 못하던 사람을 0.4에서는 탐지함

    parser.add_argument('--kpt-thr',type=float,default=0.3,help='Visualizing keypoint thresholds') 
    # 키 포인트 임계값 시각화  
    # 임계값이 너무 높으면 탐지하지 않는다..
    
    parser.add_argument('--draw-heatmap',action='store_true',
                        default=True,
                        help='Draw heatmap predicted by the model')
    # 예측 결과에 대해 heatmap 그려줄지 말지
    
    parser.add_argument('--show-kpt-idx',action='store_true',
                        default=False,
                        help='Whether to show the index of keypoints')
    # 키포인트 인덱스 표시 여부 
    
    parser.add_argument('--skeleton-style',default='mmpose',type=str,choices=['mmpose', 'openpose'],
                        help='Skeleton style selection')
    # skeleton style? mmpose or openpose ?
    
    parser.add_argument('--radius',type=int,default=3,help='Keypoint radius for visualization')
    # 시각화를 위한 키포인트 반지름
    
    parser.add_argument('--thickness',type=int,default=2,help='Link thickness for visualization')
    # 시각화를 위한 선의 두께
    
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    # 프레임당 sleep seconds??
    
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    # bbox의 투명도?
    
    parser.add_argument('--draw-bbox', action='store_true',
                        default=True, help='Draw bboxes of instances')
    # 인스턴스의 bbox?


    assert has_mmdet, 'Please install mmdet to run the demo.'
        # 조건을 만족해야함
        # 원하지 않는 조건이라면 AssertionError가 발생

    args = parser.parse_args()
    print("터미널 확인 :::::::::::::::::::: ", args) # args 확인용 추가 O

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        # print("확인용 출력 : ", output_file)
        # vis_results/demo/000001_mpiinew_test.mp4

        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
        print("\n\n터미널 확인 저장 경로 : ", args.pred_save_path )  # O

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device) 
        # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/apis/inference.py
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/utils/config_utils.py

    
    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps = args.draw_heatmap))))
    # print("터미널 확인 pose_estimator : ",pose_estimator) # 모델 구조가 출력


    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness

    # pose_estimator.cfg.visualizer.save_dir = '/home/mhncity/Desktop/SUN/code/hrnettest/mmpose/demo/vis_results' # 직접 추가. ??????t,f,경로?????????
        # /home/mhncity/anaconda3/envs/envtest/lib/python3.8/site-packages/mmengine/visualization/visualizer.py:196: 
        # UserWarning: Failed to add <class 'mmengine.visualization.vis_backend.LocalVisBackend'>, please provide the `save_dir` argument.
        # warnings.warn(f'Failed to add {vis_backend.__class__}, '
        # 이 코드가 없으면 터미널에 뜨는 문구. 이 코드를 추가해주면 이 문구 안뜸 
        # 그거없어도 잘 나오는거같음

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/registry.py

    # print("터미널 확인 psoe estimator_radius : ",pose_estimator.cfg.visualizer.radius )
    # print("터미널 확인 psoe estimator_alpha : ",pose_estimator.cfg.visualizer.alpha )
    # print("터미널 확인 psoe estimator_line_width : ",pose_estimator.cfg.visualizer.line_width )
    # print("터미널 확인 pose_estimator.cfg.visualizer : ", pose_estimator.cfg.visualizer )
        # -> pose_estimator.cfg.visualizer :  {'type': 'PoseLocalVisualizer', 
        #                                   'vis_backends': [{'type': 'LocalVisBackend', 'save_dir': None}], 
        #                                   'name': 'visualizer', 'radius': 3, 'alpha': 0.8, 'line_width': 1}
    # print("터미널 확인 visualizer : ", visualizer )
        # -> <mmpose.visualization.local_visualizer.PoseLocalVisualizer object at 0x7fb14234ba90>



    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
        # https://mmsegmentation.readthedocs.io/en/latest/api.html
        # -> set_dataset_meta(classes: Optional[List] = None, palette: Optional[List] = None, dataset_name: Optional[str] = None)
    
    

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
            # 확장자를 통해 파일의 타입을 대략적으로 추정
            # mp4 -> video 라고 알려줌
            # args.input : tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4
            # 위의 경로에서 mimetypes.guess_type(args.input)[0] 는 vidwo/mp4 
            # / 기준으로 나눈거에 0번째 인덱스 = video


    if input_type == 'image':
        # 이미지일 경우, 

        # inference ################################################################
        start= time.time()
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)
        end = time.time()
        print(f'process_one_image TIME : {end-start:.5f} sec') 

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)
                # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/structures/utils.py

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']: # 비디오니까 여기 타는중
        # 웹캠이나 비디오의 경우
        # print("터미널 확인 ::::::::: ", input_type) # O

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
                # 카메라를 통해 영상 받아오기, 동영상 갭쳐 객체 생성 
        else:
            cap = cv2.VideoCapture(args.input)
                # 동영상 갭쳐 객체 생성 
            # print("\n터미널 확인 args.input :::::", args.input) # O

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        # print("터미널 확인 save_prediction", args.save_predictions) #  true 
        print("터미널 확인 cap? ", cap)
        
        print("터미널 확인 cap.isOpened() :::: ", cap.isOpened()) # false
        print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
	    

        
        while cap.isOpened(): # => 경로문제로 false였음
                # 캡쳐 객체 초기화 확인. t or f
                # 연속해서 파일의 프레임을 읽어오기 위해 무한루프로 다음 코드 호출

            success, frame = cap.read()
                # 다음 프레임 읽기
                # 프레임을 잘 읽었다면 success는 true와 frame은 프레임 이미지, 아니라면 false와 frame는 None이 된다
                # https://bkshin.tistory.com/entry/OpenCV-3-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9E%85%EC%B6%9C%EB%A0%A5
            frame_idx += 1

            if not success:
                break

            start = time.time()
            # topdown pose estimation 
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)
                                                # args, img, detector,
                                                # pose_estimator, visualizer=None,
                                                # show_interval=0 
            end = time.time()
            print(f'TIME : {end-start:.5f} sec') 
            if args.save_predictions:
                # save prediction results

                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)
                            # /home/mhncity/Desktop/SUN/code/hrnettest/mmpose/mmpose/structures/utils.py
                            # 에 있는 split_instances 함수
                            ))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                        # video_writer 가 비어있을 경우 진행
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        # 인코딩 포맷 문자
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file, fourcc, 25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))
                        # VideoWriter 객체 생성

                video_writer.write(mmcv.rgb2bgr(frame_vis))
                    # 파일 저장

            # press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                    # cv2.waitKey(5) : 5초간 대기
                    # 0xFF == 27 : ESC
                break

            time.sleep(args.show_interval)
                # == delay time 
                # show_interval로 지정된 초 만큼 쉬고 다시 진행

        if video_writer:
            video_writer.release()
                #  비디오 객체 해제

        cap.release()
            # 비디오 출력 객체 해제

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')
            # raise : 에러 발생시키기
            # https://blockdmask.tistory.com/538
    
    if args.save_predictions:
            # save_prediction == True
        with open(args.pred_save_path, 'w') as f:
            # w = wirite 
            json.dump(
                    # dump : python dictionary 타입의 객체를 json 파일로 쓴다
                    # https://tifferent.tistory.com/40
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'\nPredictions have been saved at {args.pred_save_path}') # O


if __name__ == '__main__':
    main()
