
from pero_ocr.document_ocr.layout import PageLayout, TextLine
from pero_ocr.document_ocr.page_parser import PageParser

from typing import Tuple

import pathlib
import configparser
import cv2
import json
import numpy as np



    



class PeroOCREngine:

    def __init__(self, config_path):
        '''
        # Init the OCR pipeline. 
        # You have to specify config_path to be able to use relative paths
        # inside the config file.
        '''
        print("Init OCR pipeline...")
        config = configparser.ConfigParser()
        config.read(config_path)
        self.page_parser = PageParser(config, config_path=pathlib.Path(config_path).parent)



    def process(self, input_image: pathlib.Path, input_json: pathlib.Path, output_dir: pathlib.Path, exports: list[str]):
        """Process the column of an image using PERO OCR engine

        Args:
            input_image (pathlib.Path): Input image
            input_json (pathlib.Path): Input JSON file with COLUMN_LEVEL_1 elements
            output_dir (_type_): 
            exports (list[str]): _description_

        Returns:
            _type_: _description_
        """
        if not output_dir.is_dir():
            raise RuntimeError("Invalid output directory.")


        # Read the document page image.
        image = cv2.imread(str(input_image), 1)

        with open(input_json) as f:
            data = json.load(f)
            columns = [x for x in data if x["type"] == "COLUMN_LEVEL_1"]

        istem = input_image.stem
        output_json = list(data)
        for r in columns:
            (x,y,w,h) = r["box"]
            id = r["id"]
            ostem = f"{istem}-r{id}"
            crop = image[y:y+h, x:x+w, ...]
            # convert grayscale to color if needed
            if crop.ndim == 2:
                crop = np.tile(crop[..., np.newaxis], (1, 1, 3))
            

            # Init empty page content. 
            # This object will be updated by the ocr pipeline. id can be any string and it is used to identify the page.
            page_layout = PageLayout(id=ostem, page_size=(image.shape[0], image.shape[1]))
            # Process the image by the OCR pipeline
            page_layout = self.page_parser.process_page(crop, page_layout)


            if "page" in exports:
                page_layout.to_pagexml(str(output_dir / f"{ostem}_PAGE.xml"))
            if "alto" in exports:
                page_layout.to_altoxml(str(output_dir / f"{ostem}_ALTO.xml"))
            if "image" in exports:
                rendered_image = page_layout.render_to_image(crop) 
                cv2.imwrite(str(output_dir / f"{ostem}_debug.jpg"), rendered_image)

            for line in page_layout.lines_iterator():
                line : TextLine
                xmin = min(p[0] for p in line.polygon)
                xmax = max(p[0] for p in line.polygon)
                ymin = min(p[1] for p in line.polygon)
                ymax = max(p[1] for p in line.polygon)
                bbox = (x + xmin, y + ymax, xmax - xmin, ymax - ymin)
                e = dict(box=bbox, type="LINE", text=line.transcription, parent=id)
                output_json.append(e)

        with open(output_dir / f"{istem}.json", "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False)
        
        return output_json
            



