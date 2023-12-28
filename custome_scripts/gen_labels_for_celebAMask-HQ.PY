import os
import cv2
import glob
import numpy as np
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("path existing:", path)


def mask_to_polygans(image_path, label_idx:str):        

    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_poly = len(contours)

    lbl_str = ''
    if total_poly>0:
        for i in range(total_poly):
            if cv2.contourArea(contours[i]) > 200: #过滤很小的区域
                
                for idx, point in enumerate(contours[i]):
                    x, y = point[0]
                    x_ = x/W
                    y_ = y/H

                    if idx ==0:
                        lbl_str = label_idx + " " + lbl_str+str(x_)+" " + str(y_)
                    elif idx == len(contours[i])-1:
                        lbl_str = " " + lbl_str+str(x_)+" " + str(y_) + "\n"
                    else:
                        lbl_str = " " + lbl_str+str(x_)+" " + str(y_)
    
    return lbl_str


#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2 
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = 'CelebAMask-HQ-mask-anno'
folder_label = 'CelebAMask-HQ-label'
img_num = 30000

make_folder(folder_label)
for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    with open(os.path.join(folder_label, str(k) + '.txt'), mode='w') as f:
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if (os.path.exists(filename)):
                print (label, idx+1)
                # im = cv2.imread(filename)
                # im = im[:, :, 0]
                # im_base[im != 0] = (idx + 1)
                lbl_str = mask_to_polygans(filename, str(idx))
                # print(lbl_str)
                if lbl_str == '':
                    continue
                f.write(lbl_str)
    f.close()
        
    # filename_save = os.path.join(folder_save, str(k) + '.png')
    # print (filename_save)
    # cv2.imwrite(filename_save, im_base)

