import argparse
import glob
import json
import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml
from tqdm import tqdm
from copy import deepcopy
from utils.google_utils import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *


def attempt_download(file, repo='ultralytics/yolov5'):  # from utils.downloads import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            name = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1E5)
            return name

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except:  # fallback plan
            assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v5.0'  # current release

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def attempt_load(weights, map_location=None, fuse=True, imgsz=640, force_reexport_deployable_model=False):
    from models.models import Darknet as Model
    # from IPython import embed
    # embed()
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        _model_ckpt = ckpt['ema' if ckpt.get('ema') else 'model']
        _model = Model(opt.cfg)
        _model.load_state_dict(_model_ckpt.state_dict(), strict=True)
        # _model.nc = _model_ckpt.nc  # attach number of classes to model
        # _model.hyp = _model_ckpt.hyp  # attach hyperparameters to model
        # _model.class_weights = _model_ckpt.class_weights  # attach class weights
        # _model.names = _model_ckpt.names
        _model.to(next(_model_ckpt.parameters()).device)

        # This is the case we run validation using QAT deployable model
        if 'qat_model_quant_info' in ckpt:
            if force_reexport_deployable_model:
                from pytorch_nndct import QatProcessor
                # Image sizes
                _ori_model = deepcopy(_model)
                _model.train()
                gs = max(int(_model.stride.max()), 32)  # grid size (max stride)
                imgsz = check_img_size(imgsz, gs)  # verify imgsz is gs-multiple
                im = torch.zeros(1, 3, imgsz, imgsz).to(next(_model_ckpt.parameters()).device)  # image size(1,3,320,192) BCHW iDetection
                # dry run
                for _ in range(2):
                    y = _model(im)  # dry runs
                _model.forward = partial(_model.forward, augment=False, quant=True)
                qat_processor = QatProcessor(_model, (im,), bitwidth=8, mix_bit=False)
                calib_dir = Path(w).parent / "quantize_result"
                _trainable_model = qat_processor.trainable_model(allow_reused_module=False)
                _trainable_model.load_state_dict(ckpt['qat_ema_state_dict'], strict=True)
                _deployable_net = qat_processor.convert_to_deployable(_trainable_model, calib_dir.as_posix())
                _ori_model.load_state_dict(_deployable_net.state_dict(), strict=True)
                _model = _ori_model
            else:
                def write_quant_info(dir, quant_info_str):
                    dir.mkdir(exist_ok=True, parents=True)
                    with open(dir / 'quant_info.json', 'w') as f:
                        f.write(quant_info_str)
                w = Path(w)
                w_dir = w.parent
                write_quant_info(w_dir / "quantize_result", ckpt['qat_model_quant_info'])
        if fuse:
            model.append(_model.float().fuse().eval())  # FP32 model
        else:
            model.append(_model.float().eval())  # without layer fuse

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0,      # number of logged images
         nndct_quant=False,
         quant_mode='calib',
         dump_xmodel=False,
         nndct_bitwidth=8,
         half=True,
         rect=False,
         dump_onnx=False,
         names='data/coco.names',
         output_path='quantize_result'):  

    # Initialize/load model and set device
    training = model is not None
    do_dump = dump_xmodel or dump_onnx
    if nndct_quant:
        os.environ["W_QUANT"] = "1"
        assert half is False and augment is False, "Invalid seetings for nndct quant"
        if do_dump:
            assert quant_mode == 'test', "Quant model should be 'test' for dumping xmodel"
            assert batch_size == 1, "Dump xmodel only support batch size 1"

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # # Load model
        # model = Darknet(opt.cfg).to(device)

        # # load model
        # try:
        #     ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
        #     ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        #     model.load_state_dict(ckpt['model'], strict=False)
        # except:
        #     load_darknet_weights(model, weights[0])
        # imgsz = check_img_size(imgsz, s=64)  # check img_size

        # Load model
        # check_suffix(weights, '.pt')
        try:
            model = attempt_load(weights, map_location=device, fuse=not nndct_quant, imgsz=imgsz, force_reexport_deployable_model=not training)  # load FP32 model
        except:
            model = Darknet(opt.cfg).to(device)
            try:
                ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
                ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(ckpt['model'], strict=False)
            except:
                load_darknet_weights(model, weights[0])
        # Data
        # data = check_dataset(data)  # check
        # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=64)  # check image size


    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    # from IPython import embed
    # embed()
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=not nndct_quant)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []


    if nndct_quant:
        from pytorch_nndct.apis import torch_quantizer
        import pytorch_nndct as py_nndct
        from nndct_shared.utils import NndctOption
        from nndct_shared.base import key_names, NNDCT_KEYS, NNDCT_DEBUG_LVL, GLOBAL_MAP, NNDCT_OP
        import nndct_shared.quantization as nndct_quant
        from pytorch_nndct.quantization import torchquantizer

        input_tensor = (torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        model.forward = partial(model.forward, quant=True)
        if training:
            assert quant_mode == 'test'
            output_dir = weights
        else:
            w = Path(weights[0] if isinstance(weights, list) else weights)
            # output_dir = w.parent / "quantize_result"
            output_dir = Path(output_path)
        quantizer = torch_quantizer(quant_mode=quant_mode,
                                    bitwidth=nndct_bitwidth,
                                    module=model,
                                    input_args=input_tensor,
                                    output_dir=output_dir.as_posix())

        # TODO check the efficiency of quantization and find the reason of low efficiency
        if (NndctOption.nndct_stat.value > 2):
            def do_quantize(instance, blob, name, node=None, tensor_type='input'):
                # forward quant graph but not quantize parameter and activation
                if NndctOption.nndct_quant_off.value:
                    return blob

                blob_save = None
                if isinstance(blob.values, torch.Tensor):
                    blob_save = blob
                    blob = blob.values.data

                quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
                if blob.device.type != quant_device.type:
                    raise TypeError(
                        "Device of quantizer is {}, device of model and data should match device of quantizer".format(
                            quant_device.type))

                if (NndctOption.nndct_stat.value > 2):
                    quant_data = nndct_quant.QuantizeData(name, blob.cpu().detach().numpy())
                # quantize the tensor
                bnfp = instance.get_bnfp(name, True, tensor_type)
                if (NndctOption.nndct_stat.value > 1):
                    print('---- quant %s tensor: %s with 1/step = %g' % (
                        tensor_type, name, bnfp[1]))
                # hardware cut method
                mth = 4 if instance.lstm else 2
                if tensor_type == 'param':
                    mth = 3

                res = py_nndct.nn.NndctFixNeuron(blob,
                                                    blob,
                                                    maxamp=[bnfp[0], bnfp[1]],
                                                    method=mth)

                if (NndctOption.nndct_stat.value > 2):
                    quant_efficiency, sqnr = quant_data.quant_efficiency(blob.cpu().detach().numpy(), 8)
                    torchquantizer.global_snr_inv += 1 / sqnr
                    print(f"quant_efficiency={quant_efficiency}, global_snr_inv={torchquantizer.global_snr_inv} {quant_data._name}\n")

                # update param to nndct graph
                if tensor_type == 'param':
                    instance.update_param_to_nndct(node, name, res.cpu().detach().numpy())

                if blob_save is not None:
                    blob_save.values.data = blob
                    blob = blob_save
                    res = blob_save

                return res

            _quantizer = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
            _quantizer.do_quantize = do_quantize.__get__(_quantizer)
        ### check eficiency---end

        quant_model = quantizer.quant_model
        ori_forward = quant_model.forward
        yololayer_cfg_index = model.yololayer_num
        # module_list
        post_method144 = model.module_list[yololayer_cfg_index[0]].post_process
        post_method159 = model.module_list[yololayer_cfg_index[1]].post_process
        post_method174 = model.module_list[yololayer_cfg_index[2]].post_process
        # post_method = model.model[-1].post_process
        def forward(*args, **kwargs):
            out = ori_forward(*args, **kwargs)
            # out = out[0]
            out = list(out)
            out[0] = post_method144(out[0])
            out[1] = post_method159(out[1])
            out[2] = post_method174(out[2])
            test_out, train_out = zip(*out)
            # return post_method(out)
            return (torch.cat(test_out, 1), train_out)
        quant_model.forward = forward

    if do_dump:
        total = 1
    else:
        total = len(dataloader)

    #--------------------NNDCT QUANT------------------------------------------#


    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, total=total)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        # from IPython import embed
        # embed()
        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            if nndct_quant:
                quant_out = quant_model(img)
                inf_out, train_out = quant_out[0], quant_out[1]
            else:
                inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            # if training:  # if model has loss hyperparameters
            #     loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

        if do_dump:
            break
    # test end
    if nndct_quant and quant_mode == 'calib':
        quantizer.export_quant_config()
    if dump_xmodel:
        quantizer.export_xmodel(output_dir=output_dir.as_posix(), deploy_check=True)
        return
    if dump_onnx:
        quantizer.export_onnx_model(output_dir=output_dir.as_posix())
        return   

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # W&B logging
    if plots and wandb:
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = glob.glob('/group/dphi_algo/coco/annotations/annotations_2017/instances_val2017.json')[0]  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-csp.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='models/yolov4-csp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--dump_onnx', action='store_true', help='dump nndct onnx xmodel')
    parser.add_argument('--quant_mode', default='calib', help='nndct quant mode')
    parser.add_argument('--nndct_quant', action='store_true')
    parser.add_argument('--dump_xmodel', action='store_true', help='dump nndct xmodel')
    parser.add_argument('--nndct_stat', type=int, required=False, default=0)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--print_model', action='store_true')
    parser.add_argument('--test_rect', action='store_true')
    parser.add_argument('--output_path', default='', help='nndct quant path')

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(data=opt.data,
             weights=opt.weights,
             batch_size=opt.batch_size,
             imgsz=opt.img_size,
             conf_thres=opt.conf_thres,
             iou_thres=opt.iou_thres,
             save_json=opt.save_json,
             single_cls=opt.single_cls,
             augment=opt.augment,
             verbose=opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             nndct_quant=opt.nndct_quant,
             quant_mode=opt.quant_mode,
             dump_onnx=opt.dump_onnx,
             dump_xmodel=opt.dump_xmodel,
             half=opt.half,
             names='data/coco.names',
             output_path=opt.output_path
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov4-csp.pt', 'yolov4-csp-x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
