from enum import Enum, auto
from typing import ClassVar, List
import numpy as np
import pathlib
import json
import time
import logging
import math
from . import scribocxx
from . import scribo
from . import ner
from .pero_ocr_driver import PERO_driver

# FIXME: use ressources
street_names = pathlib.Path(__file__).parent.parent / "resources/denominations-emprises-voies-actuelles.csv"


def detect_scale(width):
    s = math.log2(2048 / width)
    rs = round(s)
    if abs(s - rs) > 0.2:
        return -1
    return int(rs)

class OCREngine(Enum):
    PERO = auto()
    TESSERACT = auto()

class OCRMode(Enum):
    LINE = auto()
    BLOCK = auto()


class App:

    def __init__(self, PERO_CONFIG_DIR, logger = None, logging_level = logging.INFO):
        if logger is None:
            self._logger = logging.Logger("scribo", level=logging_level)
            ch = logging.StreamHandler()
            ch.setLevel(logging_level)
            formatter = logging.Formatter('{asctime} - {levelname} - {message}', style="{")
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
        else:
            self._logger = logger
            self._logger.setLevel(logging_level)

        self._logging_level = self._logger.getEffectiveLevel()
        if self._logging_level == logging.DEBUG:
            scribocxx._set_debug_level(2)

        self._pero_config_dir = PERO_CONFIG_DIR
        self._pero = None



    def deskew_with_segments(self, input: np.ndarray):
        start = time.process_time_ns()
        segments = scribo.extract_segment(input)
        end = time.process_time_ns()
        self._logger.info("Segment extraction in %.1f ms", 1e-6 * (end - start))

        start = time.process_time_ns()
        angle, segments, deskewed = scribo.deskew(input, segments, scribocxx.KConfig.kAngleTolerance)
        end = time.process_time_ns()
        self._logger.info("Document deskew in %.1f ms", 1e-6 * (end - start))
        return (segments, deskewed)

    def deskew(self, input: np.ndarray):
        scale = detect_scale(input.shape[1])
        if (scale != 0 and scale != 1):
            raise RuntimeError("Invalid image dimensions. Unable to detect the scale.")

        _, deskewed = self.deskew_with_segments(input)
        return deskewed

    @staticmethod
    def objectlist_to_dictlist(segments: list):
        import pandas as pd
        data = [ { k : getattr(s, k) for k in dir(s) if not k.startswith("__")} for s in segments ]
        for d, s in zip(data, segments):
            d.update(getattr(s, "__dict__", dict()))
        return data

 


    def process_ocr(self, image: np.ndarray, regions: List[scribocxx.LayoutRegion], ocr_engine: OCREngine):
        """Run the OCR on each region of the list

        Add or update the field "text_ocr" on each textual element of the list (ENTRY, TITLE).

        Args:
            image (np.ndarray): [description]
            regions (List[scribocxx.LayoutRegion]): [description]
            ocr_engine (OCREngine): [description]
        """
        _text_categories = {scribocxx.DOMCategory.ENTRY,
            scribocxx.DOMCategory.TITLE_LEVEL_1,
            scribocxx.DOMCategory.TITLE_LEVEL_2}
        text_regions = [ r for r in regions if r.type in _text_categories ]
        try:
            self.OCR(image, text_regions, ocr_engine=ocr_engine)
        except:
            self._logger.error("OCR failed to process the following regions: %s", str(text_regions))
            for e in text_regions:
                e.text_ocr = ""

    def NER(self, texts):
        start = time.process_time_ns()
        entities = ner.detect_named_entities(texts)
        end = time.process_time_ns()
        self._logger.info("NER in %.1f ms", 1e-6 * (end - start))
        return entities

    def process_ner(self, regions: List[scribocxx.LayoutRegion]):
        """Run the NER on each ENTRY element of the list.

        Update the "ner_xml" (and some other fields) on each element of type ENTRY.

        Args:
            regions (List[scribocxx.LayoutRegion]): [description]
        """
        entries = [ r for r in regions if r.type == scribocxx.DOMCategory.ENTRY and hasattr(r, "text_ocr")]
        texts = [ e.text_ocr for e in entries ]
        entities = []
        if texts:
            entities = self.NER(texts)
        

        for x, entity in zip(entries, entities):
            x.__dict__.update(entity)



    def process(self, input: np.ndarray, font_size = 20, disable_OCR = False, disable_NER=False, ocr_engine=OCREngine.PERO):
        scale = detect_scale(input.shape[1])
        if (scale != 0 and scale != 1):
            raise RuntimeError("Invalid image dimensions. Unable to detect the scale.")

        config = scribocxx.KConfig(font_size, scale)
        segments, deskewed = self.deskew_with_segments(input)

        ## Rescaling
        subsampling_ratio = 1
        in1 = deskewed
        if (scale == 0):
            subsampling_ratio = 0.5
            in1 = scribocxx._subsample(deskewed)
            for s in segments:
                s.scale(subsampling_ratio)

        ## Backgroung substraction
        _em = font_size * subsampling_ratio
        kLineHeight = 1.5 * _em

        start = time.process_time_ns()
        debug_prefix = "debug-03" if self._logging_level == logging.DEBUG else ""
        clean = scribo.background_substraction(in1,
            kMinDiameter = 3.0 * _em,
            kMinWidth = 4.0 * _em,
            kMinHeight = 0.5 * kLineHeight,
            kMinGrayLevel = 220,
            kOpeningRadius = 1.5 * _em,
            debug_prefix=debug_prefix)
        end = time.process_time_ns()
        self._logger.info("Background suppression in %.1f ms", 1e-6 * (end - start))

        start = time.process_time_ns()
        regions = scribo.XYCutLayoutExtraction(clean, segments, config)
        end = time.process_time_ns()
        self._logger.info("Block extraction in %.1f ms", 1e-6 * (end - start))

        ## Line segmentation
        start = time.process_time_ns()
        text_block_indexes, text_blocks = zip(*[ (i, r.bbox) for i,r in enumerate(regions) if r.type  == scribocxx.DOMCategory.COLUMN_LEVEL_2])
        text_block_indexes = np.array(text_block_indexes)
        debug_prefix = "debug-05" if self._logging_level == logging.DEBUG else ""
        ws, lines = scribo.WSLineExtraction(clean, text_blocks, config, debug_prefix=debug_prefix)
        for i, l in enumerate(lines):
            l.parent_id = text_block_indexes[l.parent_id]
            l.ws_label = i + 1
        end = time.process_time_ns()
        self._logger.info("Line extraction in %.1f ms", 1e-6 * (end - start))

        ## Rescale coordinates
        if subsampling_ratio != 1:
            for r in regions:
                r.bbox.scale(1 / subsampling_ratio)
            for r in lines:
                r.bbox.scale(1 / subsampling_ratio)

        ## Extract entries (and push them in regions)
        start = time.process_time_ns()
        regions = scribo.EntryExtraction(regions, lines)
        end = time.process_time_ns()
        self._logger.info("Entry extraction in %.1f ms", 1e-6 * (end - start))


        # OCR
        if not disable_OCR:
            self.process_ocr(deskewed, regions, ocr_engine)

        # NER
        if not disable_NER:
            self.process_ner(regions)

        # Add extra tags, offset ids and sanitize
        for i, x in enumerate(regions):
            x.id = 256 + i
            x.origin = "computer"
            x.parent_id += 256

        return regions, deskewed

    def _start_pero_if_needed(self):
        if (self._pero is None):
            start_time = time.time()
            self._pero = PERO_driver(self._pero_config_dir)
            elapsed_time = int((time.time() - start_time) * 1000)
            self._logger.info(f"init 'pero ocr engine' performed in %.1f ms.", elapsed_time)

    def OCR(self, input, text_regions, ocr_engine=OCREngine.PERO, ocr_mode = OCRMode.BLOCK):
        start = time.process_time_ns()

        valid_regions = []
        for r in text_regions:
            if not r.bbox.is_valid():
                self._logger.error("Invalid region %s", r)
            else:
                valid_regions.append(r)

        if (ocr_engine is OCREngine.PERO):
            self._start_pero_if_needed()

            boxes = [(r.bbox.x0(), r.bbox.y0(), r.bbox.x1(), r.bbox.y1()) for r in valid_regions ]

            if ocr_mode == OCRMode.BLOCK:
                ocr_results = self._pero.detect_and_recognize(input, boxes)
                # for ri, result_region in enumerate(ocr_results): # DEBUG
                #     print(f"ocr_results1: region {ri}")
                #     for li, result_line in enumerate(result_region):
                #         for field in [
                #             'baseline',  # coord list, seems to be what we need
                #             # 'characters',  # charset
                #             # 'crop',  # image crop (np.array)
                #             # 'get_dense_logits',  # method
                #             # 'get_full_logprobs',  # method
                #             # 'heights',  # ?
                #             # 'id',  # id we gave it previously
                #             # 'logit_coords',  # ?
                #             # 'logits',  # ?
                #             'polygon',  # hull
                #             'transcription',  # text
                #             # 'transcription_confidence',  # None (unavail.)
                #         ]:
                #             print(f"ocr_results1: r{ri}l{li}{field}: ", getattr(result_line, field))
                #         break

                # ocr_results = [ "\n".join(textline.transcription for textline in result) for result in ocr_results ]
                # print("ocr_results2: ", ocr_results)  # DEBUG

            else:
                ocr_results = self._pero.recognize_lines(input, boxes)
                ocr_results = ocr_results.values()
                

        elif ocr_engine is OCREngine.TESSERACT: 
            boxes = [r.bbox for r in valid_regions]
            ocr_results = scribo.TesseractTextExtraction(input, boxes)

        if len(ocr_results) != len(valid_regions):
            self._logger.critical("Internal error: the number of processed boxes mismatch.")
            raise

        for e, result_region in zip(valid_regions, ocr_results):
            e.text_ocr = "\n".join(textline.transcription if hasattr(textline, "transcription") else str(textline) for textline in result_region )
            # e.line_data = {
            #     "index": result_region.index,
            #     "baseline": result_region.baseline,
            #     "polygon": result_region.polygon,
            #     "heights": result_region.heights,
            #     # "transcription": result_region.transcription,
            #     # "logits": result_region.logits,
            #     # "crop": result_region.crop,
            #     # "characters": result_region.characters,
            #     # "logit_coords": result_region.logit_coords,
            #     "transcription_confidence": result_region.transcription_confidence,
            # }

        end = time.process_time_ns()
        self._logger.info("OCR in %.1f ms", 1e-6 * (end - start))

    def OCR_lines(self, input, text_regions, ocr_engine=OCREngine.PERO):
        self.OCR(input, text_regions, ocr_engine, ocr_mode=OCRMode.LINE)



