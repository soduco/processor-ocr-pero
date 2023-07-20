from flask import abort, Blueprint, current_app, jsonify, request

from back.Application import App, OCREngine
from back.pero_ocr_driver import PERO_driver
from back import scribocxx

from storage_proxy import get_input_image

bp = Blueprint('ocr', __name__, url_prefix='/ocr')
bp.config = {}


@bp.record
def record_config(setup_state):
    bp.config = setup_state.app.config

@bp.before_request
def before_request_func():
    if request.method == "OPTIONS":
        return
    request_token = request.headers.get('Authorization')
    #if Authorization header not defined, request_token = None
    if request_token not in bp.config['TOKENS']:
        abort(403, "Invalid request token.")

# TODO add pero full page mode (do not aggregate lines from all regions)

@bp.route('/<string:ocr_engine>/regions', methods=['POST'])
def ocr_regions(ocr_engine: str):
    """Compute transcriptions for all text regions in a given document
        depending on the ocr engine used.

        Layout extraction WILL be performed for each region to detect lines
        (textual regions and lines inside them to be precise).

        The possible ocr engines are:
        - "pero"
        - "tesseract"

        Parameters must be passed as JSON post content:
        - "document" (string): document name (eg: `"Didot_1851a.pdf"`)
        - "view" (int): view to process (eg: `700`)
        - "regions" (list of regions): regions to process

        `regions` are dicts in the following format: 
        ```
        { "id": id, "bbox": [tlx, tly, brx, bry] }
        ```

        Response content is JSON payload with responses in the following format:  
        ```
        { 
            "software": "pero ocr ...",
            "content": [{"id": id_region, "lines": [ line, line ]}, ...]
        }
        ```
        where `line` has the form:
        ```
        {"id": id_line, "polygon": list of points, "transcription": text}
        ```

        Example (Python):
        ```
        engine = "tess" # or "pero"
        headers = { 'Authorization': DEBUG_TOKEN }
        query_dict = {
            "document": "Didot_1842a.pdf",
            "view": 700,
            "regions": [
                {"id": 0, "bbox": [190, 140, 590, 470]},
                {"id": 1, "bbox": [640, 910, 961, 988]},
            ]}
        response = requests.post(
            url=f"{server_uri}/ocr/{engine}/regions",
            headers=headers,
            json=query_dict)
        for region_info in response.json()["content"]:
            id_region = region_info["id"]
            lines = region_info["lines"]
            for l in lines:
                id_line = l["id"]
                polygon = l["polygon"]
                transcription = l["transcription"]
                # process results
        ```

        Example (curl):
        ```
        engine=tess # or 'pero'
        curl -X POST \
        --url https://apps.lrde.epita.fr:8000/soduco/ocr/${engine}/regions \
        --header "Authorization: $DEBUG_TOKEN" \
        --header "Content-type: application/json" \
        --data '{
            "document": "Didot_1842a.pdf",
            "view": 700,
            "regions": [
                {"id": 0, "bbox": [190, 140, 590, 470]},
                {"id": 1, "bbox": [640, 910, 961, 988]}
            ]}' \
        > result.json && jq -C < results.json
        ```
    """
    ocr_engine = getattr(OCREngine, ocr_engine.upper(), None)
    if ocr_engine is None:
        abort(400, description="Invalid ocr engine.")


    # Parse request and check
    json_data: dict = request.get_json(force=True)
    # we use `force=True` to tolerate PUT requests with 'Content-Type' header different from 'application/json'
    if json_data is None or not isinstance(json_data, dict):
        current_app.logger.info("Could not parse JSON payload")
        return "Could not parse JSON payload.", 500
    document = json_data.get("document")
    view = json_data.get("view")
    regions = json_data.get("regions")
    layout_regions = []
    errors = []
    if not isinstance(document, str):
        errors.append("Invalid required 'document' parameter (not a str or absent).")
    if not isinstance(view, int):
        errors.append("Invalid required 'view' parameter (not an int or absent).")
    if not isinstance(regions, list):
        errors.append("Invalid required 'regions' parameter (not a list or absent).")
    else:
        # Validate and parse the regions
        for ii, r in enumerate(regions):
            if len(errors) > 10:
                errors.append("(Other errors suppressed.)")
                break
            if (not "id" in r 
                or not isinstance(r["id"], (str, int))
                or not "bbox" in r
                or not isinstance(r["bbox"], (tuple, list))):
                errors.append(f"Invalid region {ii}. Expected id (int or str) and bbox (tuple or list).")
            elif len(r["bbox"]) != 4:
                errors.append(f"Invalid region {ii}. Invalid bbox (expected 4 values).")
            else:
                tlx, tly, brx, bry = r["bbox"]
                try:
                    x0, y0, w, h = int(tlx), int(tly), int(brx-tlx), int(bry-tly)
                except ValueError():
                    errors.append(f"Invalid region {ii}. Invalid bbox (expected 4 numerical values).")
                if not (0 <= x0 and 0 <= y0 and 0 < w and 0 < h):
                    errors.append(f"Invalid region {ii}. Invalid bbox (expected (tlx, tly, brx, bry) format).")
                line = scribocxx.LayoutRegion()
                # Warning: scribocxx.box constructor takes x0, y0, w, h params
                line.bbox = scribocxx.box(x0, y0, w, h)
                layout_regions.append(line)
    
    if len(errors) > 0:
        error_str = "\n".join(errors)
        current_app.logger.info(error_str)
        return error_str, 500


    # Process request
    image = get_input_image(document, view, bp.config["SODUCO_STORAGE_URI"], bp.config["SODUCO_STORAGE_AUTH_TOKEN"])
    if image is None:
        return f"Cannot find view '{view}' in document '{document}'.", 404

    # FIXME add option to chose whether we want to deskew or nor
    app = App(current_app.config["PERO_CONFIG_DIR"])
    image_deskewed = app.deskew(image)

    # FIXME should we run this in a different process to avoid freezing the server?
    app.OCR(image_deskewed, layout_regions, ocr_engine=ocr_engine)

    transcriptions_with_ids = []
    for r, reg in zip(regions, layout_regions):
        lines = []
        for l_id, line_data in enumerate(reg.text_ocr.split("\n")):
        # for l_id, l in enumerate(reg.line_data):
            line_data = {
                "id": f"{r['id']}_l{l_id}",
        #         "baseline": l.baseline.tolist(),
        #         "heights": l.heights,
        #         "polygon": l.polygon.tolist(),
                # "transcription": l.transcription
                "transcription": line_data
                }
            lines.append(line_data)
        reg_filted = {
            "id": r["id"],
            "lines": lines
            }
        transcriptions_with_ids.append(reg_filted)
        

    return jsonify({
        "content" : transcriptions_with_ids,
        "software" : PERO_driver.get_software_description()
        })


@bp.route('/<string:ocr_engine>/lines', methods=['POST'])
def ocr_lines(ocr_engine: str):
    """Compute transcriptions for all text lines in a given document
        depending on the ocr engine used.

        Layout extraction WILL NOT be performed for each line, saving time
        and producing more predictable results.

        The possible ocr engines are:
        - "pero"
        - "tesseract"

        Parameters must be passed as JSON post content:
        - "document" (string): document name (eg: `"Didot_1851a.pdf"`)
        - "view" (int): view to process (eg: `700`)
        - "lines" (list of lines): lines to process

        `lines` are dicts in the following format: 
        ```
        { "id": id, "bbox": [tlx, tly, brx, bry] }
        ```

        Response content is JSON payload with responses in the following format:  
        ```
        { 
            "software": "pero ocr ...",
            "content": [{"id": id, "transcription": text}, ...]
        }
        ```

        Example (Python):
        ```
        engine = "tess" # or "pero"
        headers = { 'Authorization': DEBUG_TOKEN }
        query_dict = {
            "document": "Didot_1842a.pdf",
            "view": 700,
            "lines": [
                {"id": 0, "bbox": [196, 145, 594, 178]},
                {"id": 1, "bbox": [196, 175, 537, 203]},
                {"id": 2, "bbox": [198, 796, 254, 825]},
            ]}
        response = requests.post(
            url=f"{server_uri}/ocr/{engine}/lines",
            headers=headers,
            json=query_dict)
        for line_info in response.json()["content"]:
            id = line_info["id"]
            transcription = line_info["transcription"]
            # process results
        ```

        Example (curl):
        ```
        engine=tess # or 'pero'
        curl -X POST \
        --url https://apps.lrde.epita.fr:8000/soduco/ocr/${engine}/lines \
        --header "Authorization: $DEBUG_TOKEN" \
        --header "Content-type: application/json" \
        --data '{
            "document": "Didot_1842a.pdf",
            "view": 700,
            "lines": [
                {"id": 0, "bbox": [196, 145, 594, 178]},
                {"id": 1, "bbox": [196, 175, 537, 203]},
                {"id": 2, "bbox": [198, 796, 254, 825]}
            ]}' \
        > result.json && jq -C < results.json
        ```
    """
    ocr_engine = getattr(OCREngine, ocr_engine.upper(), None)
    if ocr_engine is None:
        abort(400, description="Invalid ocr engine.")

    # Parse request and check
    json_data: dict = request.get_json(force=True)
    # we use `force=True` to tolerate PUT requests with 'Content-Type' header different from 'application/json'
    if json_data is None or not isinstance(json_data, dict):
        current_app.logger.info("Could not parse JSON payload")
        return "Could not parse JSON payload.", 500
    document = json_data.get("document")
    view = json_data.get("view")
    lines = json_data.get("lines")
    layout_lines = []
    errors = []
    if not isinstance(document, str):
        errors.append("Invalid required 'document' parameter (not a str or absent).")
    if not isinstance(view, int):
        errors.append("Invalid required 'view' parameter (not an int or absent).")
    if not isinstance(lines, list):
        errors.append("Invalid required 'lines' parameter (not a list or absent).")
    else:
        # Validate and parse the lines
        for ii, r in enumerate(lines):
            if len(errors) > 10:
                errors.append("(Other errors suppressed.)")
                break
            if (not "id" in r 
                or not isinstance(r["id"], (str, int))
                or not "bbox" in r
                or not isinstance(r["bbox"], (tuple, list))):
                errors.append(f"Invalid line {ii}. Expected id (int or str) and bbox (tuple or list).")
            elif len(r["bbox"]) != 4:
                errors.append(f"Invalid line {ii}. Invalid bbox (expected 4 values).")
            else:
                tlx, tly, brx, bry = r["bbox"]
                try:
                    x0, y0, w, h = int(tlx), int(tly), int(brx-tlx), int(bry-tly)
                except ValueError():
                    errors.append(f"Invalid line {ii}. Invalid bbox (expected 4 numerical values).")
                if not (0 <= x0 and 0 <= y0 and 0 < w and 0 < h):
                    errors.append(f"Invalid line {ii}. Invalid bbox (expected (tlx, tly, brx, bry) format).")
                line = scribocxx.LayoutRegion()
                # Warning: scribocxx.box constructor takes x0, y0, w, h params
                line.bbox = scribocxx.box(x0, y0, w, h)
                layout_lines.append(line)
    
    if len(errors) > 0:
        error_str = "\n".join(errors)
        current_app.logger.info(error_str)
        return error_str, 500


    # Process request
    image = get_input_image(document, view, bp.config["SODUCO_STORAGE_URI"], bp.config["SODUCO_STORAGE_AUTH_TOKEN"])
    if image is None:
        return f"Cannot find view '{view}' in document '{document}'.", 404

    # FIXME add option to chose whether we want to deskew or nor
    app = App(current_app.config["PERO_CONFIG_DIR"])
    image_deskewed = app.deskew(image)

    # FIXME should we run this in a different process to avoid freezing the server?
    app.OCR_lines(image_deskewed, layout_lines, ocr_engine=ocr_engine)

    transcriptions_with_ids = []
    for id, line in enumerate(layout_lines):
        line_result = {
            "id": id,
            "transcription": line.text_ocr
            }
        transcriptions_with_ids.append(line_result)
        
    return jsonify({
        "content" : transcriptions_with_ids,
        "software" : PERO_driver.get_software_description()
        })

