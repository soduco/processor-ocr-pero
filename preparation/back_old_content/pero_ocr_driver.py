
import configparser
from functools import lru_cache
import os

import numpy as np
import cv2

from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
# from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR


@lru_cache(maxsize=1)
def _load_pero_page_parser(config_path):
    config = configparser.ConfigParser()
    config_file = os.path.join(config_path, "config_cpu.ini")
    if not os.path.exists(config_file):
        raise ValueError(f"cannot read configuration file {config_file}")
    config.read(config_file)
    return PageParser(config, config_path)


class PERO_driver():
    def __init__(self, config_path: str) -> None:
        """
        Wrapper to PERO OCR.

        Args:
            config_path (str): Path to configuration dir.
                It must contain the following files:
                - ParseNet.pb
                - checkpoint_350000.pth
                - config.ini
                - ocr_engine.json
        """
        self.config_path = config_path
        self.page_parser = _load_pero_page_parser(config_path)

        # Reuse already initialized OCR engine
        self.ocr_engine = self.page_parser.ocr.ocr_engine


    @staticmethod
    def get_software_description():
        # ideally we would have some UID here which points to a single db with all parameters and weights to reproduce.
        return "Pero OCR v2021-11-23 github master branch, models: pero_eu_cz_print_newspapers_2020-10-07"


    def detect_and_recognize(self, image, bbox_list: list) -> list:
        """Process rectangular regions by detecting text regions and lines, then OCRing them.

        Args:
            image (np.ndarray): Full image to crop regions from
            bbox_list (list of tuples of int): bounding boxes of the regions

        Returns:
            list of Pero lines: List of complex line objects are produced by Pero
        """
        # This should run in a different thread / process / worker machine to avoid freezing the server
        line_lists = []  # list of list of lines (a list of lines for each region)
        for (tlx, tly, blx, bly) in bbox_list:
            if not (0 < tlx < blx and 0 < tly < bly):
                # skipping invalid bbox
                line_lists.append([])
                continue
            crop = image[tly:bly, tlx:blx, ...]
            if crop.ndim == 2:
                # convert grayscale to color if needed
                crop = np.tile(crop[..., np.newaxis], (1, 1, 3))

            page_layout = PageLayout(id="00", page_size=(crop.shape[0], crop.shape[1]))

            # The real thing
            page_layout2 = self.page_parser.process_page(crop, page_layout)

            line_lists.append(list(page_layout2.lines_iterator()))
        return line_lists


    def recognize_lines(self, image, bbox_list: list, pad_color=255) -> dict:
        """Process small rectangular regions to recognize the text contained in each
        of them.

        Layout analysis is NOT run here, it is the responsability of the caller to 
        ensure that each region contains only a single line of text.

        Args:
            image (np.ndarray): Full image to crop lines from
            bbox_list (list of tuples of int): bounding boxes of the lines
            pad_color (tuple of int or int): value to use as background when padding
                lines

        Returns:
            dict[int, str]: Mapping of original bbox id to transcription (if any).
                This allows to skip bad lines.
        """
        # This should run in a different thread / process / worker machine to avoid freezing the server
        
        # Gather cropped images in RGB format
        crops_list = []  # list of list of lines (a list of lines for each region)
        orig_idx = []  # list of original bbox query for each valid crop
        for ii, (tlx, tly, blx, bly) in enumerate(bbox_list):
            if not (0 < tlx < blx and 0 < tly < bly):
                # skipping invalid bbox
                continue
            crop = image[tly:bly, tlx:blx, ...]
            if crop.ndim == 2:
                # convert grayscale to color if needed
                crop = np.tile(crop[..., np.newaxis], (1, 1, 3))
            crops_list.append(crop)
            orig_idx.append(ii)
        
        # Prepare the images so they all have the same shape
        target_h = self.ocr_engine.line_px_height
        max_width = self.ocr_engine.max_input_horizontal_pixels
        crops_ready = self.resize_and_pad_images(crops_list, target_h=target_h, max_width=max_width, bg_color=pad_color)

        # Real thing here
        all_transcriptions, _all_logits, _all_logit_coords = self.ocr_engine.process_lines(crops_ready)

        # Return mapping of {valid_idx -> transcription}
        results = {}
        for oi, tr in zip(orig_idx, all_transcriptions):
            results[oi] = tr
        return results

    @staticmethod
    def resize_and_pad_images(img_list, target_h, max_width, bg_color=255):
        '''
        Generates a new list where all images have the same shape.
        
        Will use resizing to reduce if needed, and padding otherwise.
        Background is filled with `bg_color` when padding.
        
        The target shape is `(target_h, min(W, max_width), channels)` where:
        - `target_h` is a given parameter
        - `max_width` is a given parameter
        - `W` is the width of the largest image (after scaling)
        - `channels` is the number of channels (all images must be either RGB or grayscale)
        '''
        # We expect at least 1 element
        if len(img_list) == 0:
            return []
        img0 = img_list[0]
        
        # Gather shapes
        shapes = np.array([img.shape[:2] for img in img_list])
        
        # Check channels
        if img_list[0].ndim > 2:
            # color case
            if not all(np.array([img.shape[2] for img in img_list]) == img0.shape[2]):
                raise ValueError(f"All images must have the same number of channels.")
        
        # Compute target shapes
        resized_shapes = []
        for ii, (h, w) in enumerate(shapes):
            ratio = 1.
            new_h = h
            new_w = w
            if new_h > target_h:
                # Try to resize image
                new_h = target_h
                ratio = target_h / h
                new_w = int(w * ratio)
            
            if new_w > max_width:
                print(f"WARNING: large image (f{ii}): width after resize is f{new_w} and max width is f{max_width}.")
                new_w = max_width
                ratio = max_width / w
                new_h = int(h * ratio)
        
            resized_shapes.append((new_h, new_w))
        
            # Check for small images
            if h < target_h / 2:
                print(f"WARNING: small image (f{ii}): target height is f{target_h} but image height is f{h}.")
            if w < target_h:
                print(f"WARNING: thin image (f{ii}): image height is f{w} (target height is f{target_h} and max width is f{max_width}.")

        # print("\n".join([f"({h1}, {w1}) > ({h2}, {w2})" for (h1, w1), (h2, w2) in zip(shapes, resized_shapes)]))
        
        # Need actual_max_w for padding
        actual_max_w = np.max(np.array(resized_shapes)[:,1])
        # print(f"actual_max_w: {actual_max_w}")
        
        # Resize and pad images
        final_shape = (target_h, actual_max_w, 3) if img0.ndim > 2 else (target_h, actual_max_w)
        final_images = []
        for (new_h, new_w), img in zip(resized_shapes, img_list):
            h, w = img.shape[:2]
            
            resized = img
            if new_h < h:
                # Must resize
                resized = cv2.resize(img, (new_w, new_h))
            
            padded = resized
            if new_h < target_h or new_w < actual_max_w:
                # Must pad
                padded = np.full(final_shape, bg_color, dtype=img.dtype)
                padded[:resized.shape[0],:resized.shape[1],...] = resized
            
            final_images.append(padded)
            
        return final_images

