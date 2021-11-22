import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import divyam_devansh


@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           image_to_be_classified=cv2.imread("pic1.PNG"),
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.3,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold  #0.45
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=True,  # show results
           save_txt=True,  # save results to *.txt
           save_conf=True,  # save confidences in --save-txt labels
           save_crop=True,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=2,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
    # imgsz=image_to_be_classified.shape[0]
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # print(image_to_be_classified.shape[0])
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # imgsz=image_to_be_classified.shape[0]
    # print("dheere dheere=",imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    stop_video=0
    cnt = 0

    img,im0s=divyam_devansh(image_to_be_classified,img_size=imgsz,stride=stride)
    for i in range(1):
        #cv2.imshow("mast magan",im0s.shape)
        # cv2.waitKey(10000)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()
        # print(conf_thres,iou_thres,classes,agnostic_nms,max_det)
        # print(t1)

        # Apply Classifier

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # print(pred)

        # Process detections
        # Every Single frame of the video
        boundingbox=np.array([0,0,0,0,"0",0])
        # return boundingbox
        # print("hi guys")
        for i, det in enumerate(pred):  # detections per image
            # print("gggg")
            # #previous code:
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            # else:
            #     p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # if webcam:  # batch_size >= 1
            #     s, im0, frame =  f'{i}: ', im0s[i].copy(), dataset.count
            # else:
            s, im0=  '', im0s.copy()
            # cv2.imshow("im0", im0)
            # im0=frame
            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # print(len(det))
            if len(det):

                # print("ffffffffffff")
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # print("Ggggggg")
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #Every single detection in a particular frame
                coordinates=[]
                for *xyxy, conf, cls in reversed(det):

                    # print("Ggggggg")
                    # if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    label=label.split(" ")
                    addrow= np.array([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), label[0], label[1]])
                    boundingbox=np.vstack((boundingbox,addrow))
                    # print("hello guys chai peeelo",boundingbox)
                # print("shape",im0.shape)
                return 1,boundingbox,im0
            else:
                return 0,boundingbox,im0
            #Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cnt=cnt+1
                # print(im0.shape)
                cv2.imshow("str(p)", im0)
                key=cv2.waitKey(10000)  # 1 millisecond
                if(key==ord("d")):
                    stop_video=1

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)
        if(stop_video==1):
            break



    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')

#
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images/bus.jpg', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img',default=True, action='store_true', help='show results')
    # parser.add_argument('--save-txt',default=True, action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf',default=True, action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop',default=True, action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave',action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', default=2,nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # opt = parser.parse_args()
    # # print(opt)
    # check_requirements(exclude=('tensorboard', 'thop'))
    #
    # detect(**vars(opt))
    detect()