import tensorflow as tf
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.engine.results import Results
import cv2
import numpy as np
import torch



def preprocess(im, im_size = 640):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        if im.shape != (im_size, im_size, 3):
            im = cv2.resize(im, (im_size, im_size))
        im = tf.convert_to_tensor(np.array([im]), dtype=tf.float32)
        im = im/255.0
        return im

def tflite_run(batch_img):
    model = tf.lite.Interpreter(model_path="runs/segment/train18/weights/best_saved_model/best_int8.tflite")

    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()
    # print(input_details)
    # print(output_details)
    model.set_tensor(input_details[0]["index"], batch_img)
    model.invoke()
    out_1 = model.get_tensor(output_details[0]['index'])
    out_2 = model.get_tensor(output_details[1]['index']) #(160,160,32)

    return [out_1, out_2]

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, scores, threshold):
    """
    简单的非极大值抑制算法
    Args:
    - boxes: 框的坐标，每行为一个框的 [x1, y1, x2, y2]
    - scores: 对应框的置信度得分
    - threshold: IoU 阈值，用于判断两个框是否重叠

    Returns:
    - selected_indices: 选中的框的索引
    """
    selected_indices = []

    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    while len(sorted_indices) > 0:
        best_index = sorted_indices[0]
        selected_indices.append(best_index)

        iou_values = []
        for index in sorted_indices[1:]:
            iou = calculate_iou(boxes[best_index], boxes[index])
            iou_values.append(iou)

        # 移除与选中框重叠度大于阈值的框
        indices_to_remove = [i + 1 for i, iou in enumerate(iou_values) if iou > threshold]
        sorted_indices = [i for j, i in enumerate(sorted_indices) if j not in indices_to_remove]

    return selected_indices



def xywh2xyxy(bbox_xywh, shape_hw):
    if len(bbox_xywh.shape) == 1:
        dw = bbox_xywh[2] / 2  # half-width
        dh = bbox_xywh[3] / 2  # half-height
        x1 = int((bbox_xywh[0] - dw)*shape_hw[1])  # top left x
        y1 = int((bbox_xywh[1] - dh)*shape_hw[0])   # top left y
        x2 = int((bbox_xywh[0] + dw)*shape_hw[1])   # bottom right x
        y2 = int((bbox_xywh[1] + dh)*shape_hw[0])   # bottom right y
        return (x1, y1, x2, y2)
    elif len(bbox_xywh.shape) == 2:
        dw = bbox_xywh[:, 2] / 2  # half-width
        dh = bbox_xywh[:, 3] / 2  # half-height
        x1 = int((bbox_xywh[:, 0] - dw)*shape_hw[1])  # top left x
        y1 = int((bbox_xywh[:, 1] - dh)*shape_hw[0])   # top left y
        x2 = int((bbox_xywh[:, 0] + dw)*shape_hw[1])   # bottom right x
        y2 = int((bbox_xywh[:, 1] + dh)*shape_hw[0])   # bottom right y
        return (x1, y1, x2, y2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gen_box_mask(s_mask, bbox, shape, resize_to_input_shape = True):

    f_mask = np.zeros_like(s_mask)
    mh, mw = f_mask.shape[1:3]
    for idx, bx in enumerate(bbox):
        x1, y1, x2, y2 = xywh2xyxy(bx, (mh, mw))
        f_mask[idx, y1:y2, x1:x2] = s_mask[idx, y1:y2, x1:x2]
    return f_mask


def gen_mask(proto, mask_in, bbox, shape, f_featuremap=(160,160), mask_thre = 0.5):
    
    mask_tmp =  mask_in@(proto.transpose(-1, 0, 1).reshape(32, -1))   
    mask = sigmoid(mask_tmp).reshape(mask_in.shape[0], f_featuremap[0],f_featuremap[1])
    s_mask = np.where(mask>mask_thre, 255.0, 0.0)
    f_mask = gen_box_mask(s_mask, bbox, shape)

    # cv2.imwrite("m1.jpg", f_mask[0, ...])
    # cv2.imwrite("m2.jpg", f_mask[1, ...])
    # cv2.imwrite("m3.jpg", f_mask[2, ...])
    return f_mask
def postprocess(preds,  img, conf_thres = 0.8, ):
    # mask_head = preds[1] #(bs, mask, m_w, m_h)

    obj_head = preds[0] #(bs, xywh_conf_class_mask, anchor)  xywh:中心点 + 框的宽高
    transposed = obj_head[0].transpose(-1, -2)

    conf = transposed[:, 4:22]
    conf_idx, class_idxes = np.where(conf>conf_thres)
    # print(conf_idx)
    # print(class_idxes)
    all_dected_cls = np.unique(class_idxes) #找出相同的类别， 每个类型单独使用nms， 获取最终的侯选框
    single_class_idxes = []
    single_conf_idxes = []
    for cls_idx in all_dected_cls:
        tmp_cls = []
        conf_idxes = []
        for j in range(len(class_idxes)):
            if cls_idx == class_idxes[j]:
                tmp_cls.append(conf_idx[j])
                conf_idxes.append(cls_idx)
        single_class_idxes.append(tmp_cls) 
        single_conf_idxes.append(conf_idxes)      

    final_filtered = []
    for idx, sc_idx in enumerate(single_class_idxes):
        single_class_filterd = transposed[sc_idx]
        tmp_srores = single_class_filterd[:, 4:22][:, single_conf_idxes[idx][0]]
        
        f_tmp_idx = tmp_srores.argmax()
        final_filtered.append(sc_idx[f_tmp_idx])
        
        # f_tmp_idx = non_max_suppression()
    fileted = transposed[final_filtered, ...]
    
    if fileted.shape[0]>0:

        f_mask = gen_mask(preds[1][0], fileted[:, 22:], fileted[:, 0:4], img.shape[0:2], f_featuremap=(128,128))
        return f_mask
    else:
        return np.zeros((1, 128,128))
    
    # # fileted = transposed[conf_idx]   
    # h, w = img.shape[0:2]
    # final_filtered = []

    # for bbox_in in fileted:
        
    #     xy_wh = bbox_in[0:4]
    #     print(xy_wh)
    #     color = (0, 255, 0)  # BGR颜色，这里是绿色
    #     x1, y1, x2, y2= xywh2xyxy(xy_wh, (h, w))
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # cv2.imshow("", img)
    # cv2.waitKey(0)



def run():
    camera_no = 0
    cap = cv2.VideoCapture(camera_no)
    
    while cap.isOpened():
        res, frame = cap.read()
        if res:
            im = preprocess(frame, im_size=512)
            preds = tflite_run(im)
            masks = postprocess(preds, img = frame)

            cv2.imshow(winname="YOLOV8", mat=masks[0, ...])
    
            if cv2.waitKey(1) == 27:
                break
    
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows() 



def demo_run():
    frame =cv2.imread('t.jpg')
    im = preprocess(frame)
    preds = tflite_run(im)
    postprocess(preds, img = frame)


if __name__ == "__main__":
    # demo_run()
    run()