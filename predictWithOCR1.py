import hydra
import torch
import easyocr
import cv2
import csv
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import re

def validate_ocr_text(ocr_text):
    pattern1 = r"^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$"  # AB12DC3456
    pattern2 = r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$"    # AB1DCE3456
    return re.match(pattern1, ocr_text) or re.match(pattern2, ocr_text)

# def getOCR(im, coors):
#     x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
#     im = im[y:h, x:w]
#     conf = 0.2

#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     results = reader.readtext(gray)
#     ocr = ""
#     ocr_confidence = 0.0

#     for result in results:
#         if len(results) == 1:
#             ocr = result[1]
#             ocr_confidence = result[2]
#         if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
#             ocr = result[1]
#             ocr_confidence = result[2]

#     # Debugging print statement
#     print(f"OCR Result: {ocr}, Confidence: {ocr_confidence}")
    
#     return str(ocr), ocr_confidence

def getOCR(im, coors):
    try:
        x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
        im = im[y:h, x:w]  # Crop the image to the bounding box
        conf = 0.2

        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(gray)
        ocr, ocr_confidence = "", 0.0

        for result in results:
            text = result[1].replace(" ", "").upper()  # Clean and normalize the OCR text
            if validate_ocr_text(text):  # Validate against the regex patterns
                ocr, ocr_confidence = text, result[2]
                break  # Stop after finding the first valid match

        # Debugging print statement
        print(f"OCR Result: {ocr}, Confidence: {ocr_confidence}")
        return ocr, ocr_confidence

    except Exception as e:
        print(f"Error in getOCR: {e}")
        return "", 0.0  # Return default values if any error occurs


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    # def write_results(self, idx, preds, batch):
    #     p, im, im0 = batch
    #     log_string = ""
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    #     self.seen += 1
    #     im0 = im0.copy()
    #     if self.webcam:  # batch_size >= 1
    #         log_string += f'{idx}: '
    #         frame = self.dataset.count
    #     else:
    #         frame = getattr(self.dataset, 'frame', 0)

    #     self.data_path = p
    #     self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
    #     log_string += '%gx%g ' % im.shape[2:]  # print string
    #     self.annotator = self.get_annotator(im0)

    #     det = preds[idx]
    #     self.all_outputs.append(det)
    #     if len(det) == 0:
    #         return log_string

    #     with open('results.csv', 'a', newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerow(['Image', 'OCR Text', 'Confidence'])

    #         for c in det[:, 5].unique():
    #             n = (det[:, 5] == c).sum()  # detections per class
    #             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    #         # write
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         for *xyxy, conf, cls in reversed(det):
    #             if self.args.save_txt:  # Write to file
    #                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                 line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
    #                 with open(f'{self.txt_path}.txt', 'a') as f:
    #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

    #             if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
    #                 c = int(cls)  # integer class
    #                 label = None if self.args.hide_labels else (
    #                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
    #                 ocr, ocr_confidence = getOCR(im0, xyxy)
    #                 if ocr != "":
    #                     label = ocr
    #                 self.annotator.box_label(xyxy, label, color=colors(c, True))
    #                 # Debugging print statement
    #                 print(f"Image: {self.data_path.stem}, OCR: {ocr}, Confidence: {ocr_confidence}")
    #                 csvwriter.writerow([self.data_path.stem, ocr, ocr_confidence])
    #             if self.args.save_crop:
    #                 imc = im0.copy()
    #                 save_one_box(xyxy,
    #                              imc,
    #                              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
    #                              BGR=True)

    #     return log_string
    def write_results(self, idx, preds, batch):
      p, im, im0 = batch
      log_string = ""
      if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
      self.seen += 1
      im0 = im0.copy()
      frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)

      self.data_path = p
      self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
      log_string += '%gx%g ' % im.shape[2:]  # print string
      self.annotator = self.get_annotator(im0)

      det = preds[idx]
      self.all_outputs.append(det)
      if len(det) == 0:
        return log_string

    # Write CSV header only once
      if not hasattr(self, 'csv_header_written'):
        with open('results.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image', 'OCR Text', 'Confidence'])
        self.csv_header_written = True

      with open('results.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = None if self.args.hide_labels else (
                self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
            ocr, ocr_confidence = getOCR(im0, xyxy)
            if validate_ocr_text(ocr) and ocr_confidence > 0.5:  # Only store validated text
                label = ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
                print(f"Image: {self.data_path.stem}, OCR: {ocr}, Confidence: {ocr_confidence}")
                csvwriter.writerow([self.data_path.stem, ocr, ocr_confidence])

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()