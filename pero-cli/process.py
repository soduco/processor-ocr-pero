
from pero_ocr.document_ocr.layout import PageLayout, TextLine
from pero_ocr.document_ocr.page_parser import PageParser

from typing import Tuple, List, Any

import pathlib
import configparser
import cv2
import json
import numpy as np

def add_margin(lines : List[Any], column_box: Tuple[int,int,int,int], normalize=True):
    '''
    Take a list of lines and add the left/right indent (relative and absolute) value to each LINE
    If normalize, the indentation value is normalized with the column width
    '''
    if len(lines) == 0:
        return

    (x, _, w, _) = column_box
    for e in lines:
        (x0, _, w0, _) = e["box"]
        left = np.clip(x0 - x, 0, w)
        right = np.clip((x + w) - (x0 + w0), 0, w)
        if normalize:
            left = left / w
            right = right / w
        e["margin-left"] = left
        e["margin-right"] = right


    
    cline = lines[0]
    pml, pmr = cline["margin-left"],  cline["margin-right"]
    cline["margin-left-relative"] = 0
    cline["margin-right-relative"] = 0
    if len(lines) > 1:
        for cline in lines[1:]:
            cml, cmr = cline["margin-left"],  cline["margin-right"]
            cline["margin-left-relative"] = cml - pml
            cline["margin-right-relative"] = cmr - pmr
            pml, pmr = cml, cmr


    



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
        dir = input_json.parent.name
        page = input_json.stem


        if not output_dir.is_dir():
            raise RuntimeError("Invalid output directory.")

        for ex in exports:
            (output_dir / ex / dir).mkdir(exist_ok=True, parents=True)


        # Read the document page image.
        image = cv2.imread(str(input_image), 1)

        with open(input_json) as f:
            data = json.load(f)
            columns = [x for x in data if x["type"] == "COLUMN_LEVEL_1"]

        # Remove previously detected lines
        data = filter(lambda x: x["type"] != "LINE", data)


        output_json = list(data)
        for r in columns:
            (x,y,w,h) = r["box"]
            id = r["id"]

            crop = image[y:y+h, x:x+w, ...]
            # convert grayscale to color if needed
            if crop.ndim == 2:
                crop = np.tile(crop[..., np.newaxis], (1, 1, 3))


            # Init empty page content. 
            # This object will be updated by the ocr pipeline. id can be any string and it is used to identify the page.
            page_layout = PageLayout(id=f"{dir}/{page}", page_size=(image.shape[0], image.shape[1]))
            # Process the image by the OCR pipeline
            page_layout = self.page_parser.process_page(crop, page_layout)


            if "page" in exports:
                page_layout.to_pagexml(str(output_dir / f"page/{dir}/{page}.xml"))
            if "alto" in exports:
                page_layout.to_altoxml(str(output_dir / f"alto/{dir}/{page}.xml"))
            if "image" in exports:
                rendered_image = page_layout.render_to_image(crop)
                cv2.imwrite(str(output_dir / f"image/{dir}/{page}.jpg"), rendered_image)

            lines = []
            start = 10000 
            for i, line in enumerate(page_layout.lines_iterator()):
                line : TextLine
                xmin = min(p[0] for p in line.polygon)
                xmax = max(p[0] for p in line.polygon)
                ymin = min(p[1] for p in line.polygon)
                ymax = max(p[1] for p in line.polygon)
                bbox = (x + xmin, y + ymin, xmax - xmin, ymax - ymin)
                e = dict(box=bbox, type="LINE", text=line.transcription, parent=id, id=start + i)
                lines.append(e)
                output_json.append(e)
            add_margin(lines, r["box"])

        with open(output_dir / f"json/{dir}/{page}.json", "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False)

        return output_json




