import os
import numpy as np
import cv2

class Custome(object):
    def __init__(self, img_dir, label_dir, dst_label_dir = "") -> None:
        self.image_dir = img_dir
        self.label_dir = label_dir
        if dst_label_dir=="":
            self.dst_label_dir = label_dir+"_label"
        else:
             self.dst_label_dir = dst_label_dir
        if not os.path.exists(self.dst_label_dir):
            os.makedirs(self.dst_label_dir)

        self.numclasses = 19
        self.label_paths = os.listdir(self.label_dir)

        super().__init__()

    def break_down(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True
        return False

    def gen_label(self):
        for f in self.label_paths:
            print(f)
            img = cv2.imread(os.path.join(self.label_dir, f), cv2.IMREAD_GRAYSCALE)
            dst_labl_path = os.path.join(self.dst_label_dir, f.replace(".png", ".txt"))
            h, w = img.shape            
            label_line = ""
            for label in range(self.numclasses):
                label_line = label_line+str(label)
                for r in range(h):
                    for c in range(w):
                        if img[r,c]==label:
                            r_cord = float(r)/float(h)
                            c_cord = float(c)/float(w)
                            label_line = label_line + " " + "{:.6f}".format(r_cord)+" {:.6f}".format(c_cord)
                        else:
                            continue
                label_line = label_line+"\n"

           
            with open(dst_labl_path, mode="w") as f:
                f.write(label_line)
            f.close()

            if self.break_down:
                break


    def call(self):
        # print(self.label_paths)
        self.gen_label()


    def mask_to_polygans(self):
        

        image_path = "/home/zhangp/3TB/deep_detection/CelebAMask-HQ/face_parsing/Data_preprocessing/CelebAMask-HQ-mask-anno/0/00000_hair.png"
        # load the binary mask and get its contours
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            # print(cv2.contourArea(cnt))
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

            # # print the polygons
            # with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
            #     for polygon in polygons:
            #         for p_, p in enumerate(polygon):
            #             if p_ == len(polygon) - 1:
            #                 f.write('{}\n'.format(p))
            #             elif p_ == 0:
            #                 f.write('0 {} '.format(p))
            #             else:
            #                 f.write('{} '.format(p))

            #     f.close()
                
                

if __name__=="__main__":
    imgdir = "/home/zhangp/3TB/deep_detection/CelebAMask-HQ/face_parsing/Data_preprocessing/test_img"
    labeldir = "/home/zhangp/3TB/deep_detection/CelebAMask-HQ/face_parsing/Data_preprocessing/test_label"
    ct = Custome(img_dir=imgdir, label_dir=labeldir)
    ct.mask_to_polygans()


    # a = 3.1415
    # b = "{:.3f}".format(a)
    # print(b)