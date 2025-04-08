import base64
from pathlib import Path
from typing import Any

import cv2
import modal
from modal import Image, Volume
from promptstore import PromptStore

from corral.agents.utils import (
    LiteLLMMessage,
    llm_call,
)
from corral.utils import (
    MODAL_TOOL_REGISTRY,
    modal_tool,
    tool,
    vector_database_search,
    web_search,
)

current_file_dir = Path(__file__).parent
store = PromptStore(current_file_dir / "prompts")
app = modal.App("macbench")
hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)


@tool
def enhanced_brave_search(
    query: str, num_results: int = 5, min_similarity: float = 0.75
) -> list[dict]:
    """Perform a web search using Brave Search, then filter and rank results using embeddings.

    Args:
        query (str): The search query string
        num_results (int, optional): Maximum number of results to return. Defaults to 5
        min_similarity (float, optional): Minimum similarity score threshold. Defaults to 0.75

    Returns:
        list[dict]: A list of dictionaries containing the most relevant search results
        with their content and metadata, sorted by similarity score

    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    return web_search(
        query=query,
        num_results=num_results,
        min_similarity=min_similarity,
    )


@tool
def search_lab_safety(query: str) -> list[dict]:
    """Search lab safety knowledge database for relevant information or guidelines.

    Args:
        query (str): The search query related to laboratory safety

    Returns:
        list[dict]: A list of lab safety information entries with content and relevance scores

    Raises:
        RuntimeError: If the lab safety collection doesn't exist
    """
    return vector_database_search(query, collection_name="lab_safety_collection")


@tool
def search_ms_guide(query: str) -> list[dict]:
    """Search mass spectrometry knowledge database for relevant information or guidelines.

    Args:
        query (str): The search query related to mass spectrometry

    Returns:
        list[dict]: A list of mass spectrometry information entries with content and relevance scores

    Raises:
        RuntimeError: If the mass spectrometry collection doesn't exist
    """
    return vector_database_search(query, collection_name="ms_guide_collection")


@tool
def search_nmr_guide(query: str) -> list[dict]:
    """Search NMR spectroscopy knowledge database for relevant information or guidelines.

    Args:
        query (str): The search query related to NMR spectroscopy

    Returns:
        list[dict]: A list of NMR spectroscopy information entries with content and relevance scores

    Raises:
        RuntimeError: If the NMR spectroscopy collection doesn't exist
    """
    return vector_database_search(query, collection_name="nmr_guide_collection")


@tool
def llm_vision_expert(query: str, image_path: str) -> str:
    """
    Expert in extracting information from charts, plots and tables using a VLLM.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        query (str): The question or instruction related to the image
        image_path (str): The path to the image file

    Returns:
        str: The response from the LLM regarding the image analysis
    """
    # Ideally we would like to use the latest model
    # This Gemini seems to be the best one for OCR tasks
    model = "gemini/gemini-2.5-pro-preview-03-25"

    try:
        with Path(image_path).open("rb") as f:
            image_bytes = f.read()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        system_prompt = store.get("9c471e7f-7bbd-4ef4-8068-bf66e838e590")
        system_prompt = system_prompt.fill({})
        user_prompt = [
            {
                "type": "text",
                "text": f"Query: {query}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            },
        ]
        messages = [
            LiteLLMMessage(role="system", content=system_prompt),
            LiteLLMMessage(role="user", content=user_prompt),
        ]
        # str to avoid typing warnings
        return str(
            llm_call(
                model=model,
                messages=messages,
                temperature=0.0,
            )
        )
    except Exception as e:
        return f"Error while extracting data: {e}"


@modal_tool(
    app=app,
    image=Image.debian_slim().pip_install(
        "transformers", "Pillow", "torch", "torchvision", "loguru"
    ),
    gpu="A100-40GB",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def deplot_image_extractor_modal(image_bytes: bytes) -> str:
    """
    Extract data from charts and plots using Google's Deplot model.

    Args:
        image_bytes (bytes): Image file data as bytes

    Returns:
        str: String containing the extracted data table representation
    """
    import io

    from loguru import logger
    from PIL import Image
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    logger.info("Starting deplot_image_extractor_modal function")
    try:
        logger.info("Loading Pix2Struct processor and model")
        processor = Pix2StructProcessor.from_pretrained("google/deplot")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
        logger.info("Successfully loaded processor and model")

        logger.debug(f"Processing image of size {len(image_bytes)} bytes")
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(
            f"Image loaded successfully: {image.format} image, size {image.size}"
        )

        logger.info("Preparing inputs for model")
        inputs = processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        )
        logger.info("Running model inference")
        predictions = model.generate(**inputs)
        logger.info("Inference complete, decoding results")

        result = str(processor.decode(predictions[0], skip_special_tokens=True))
        logger.info(
            f"Successfully extracted data from image (output length: {len(result)} chars)"
        )
        return result
    except Exception as e:
        logger.error(f"Error in deplot_image_extractor_modal: {e}")
        return f"Error while extracting data: {e}"


deplot_image_extractor_remote = MODAL_TOOL_REGISTRY["deplot_image_extractor_modal"]


@tool
def deplot_image_extractor(image_path: str) -> str:
    """
    Extract data from charts and plots using Google's Deplot model.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path (str): Path to the image file containing a chart or plot

    Returns:
        str: String containing the extracted data table representation
    """
    from PIL import Image

    with Image.open(image_path) as img:
        import io

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        img_bytes = img_byte_arr.getvalue()

    return deplot_image_extractor_remote.remote(img_bytes)


@modal_tool(
    app=app,
    image=Image.debian_slim()
    .apt_install(
        "tesseract-ocr",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .pip_install("pytesseract", "opencv-python", "Pillow", "numpy", "loguru"),
    gpu="A100-40GB",
    timeout=600,
)
def extract_table_text_modal(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from images containing tables using OCR (pyTesseract).

    Args:
        image_bytes (bytes): Image file data as bytes
        lang (str, optional): Language code for OCR. Default: 'eng' for English

    Returns:
        str: String containing the extracted text from the table image
    """
    import io

    import cv2
    import numpy as np
    import pytesseract
    from loguru import logger
    from PIL import Image

    logger.info(f"Starting OCR processing with language: {lang}")
    logger.debug(f"Received image data of size: {len(image_bytes)} bytes")

    image = Image.open(io.BytesIO(image_bytes))
    logger.info(f"Loaded image with dimensions: {image.size}")

    # Preprocess the image for OCR
    logger.info("Starting image preprocessing")
    # Convert to grayscale and apply thresholding
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.debug("Converted image to grayscale")

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    logger.debug("Applied thresholding to image")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    logger.debug("Applied morphological operations")

    custom_config = r"--oem 3 --psm 6 -l " + lang
    logger.info(f"OCR configuration: {custom_config}")

    logger.info("Extracting raw text with pytesseract")
    extracted_text = pytesseract.image_to_string(processed, config=custom_config)
    logger.debug(f"Raw text extraction complete: {len(extracted_text)} characters")

    try:
        logger.info("Extracting structured table data")
        table_data = pytesseract.image_to_data(
            processed, config=custom_config, output_type=pytesseract.Output.DICT
        )

        logger.debug(f"Found {len(table_data['text'])} text elements in table")
        structured_text = "Structured Table Data:\n"
        last_block_num = -1
        text_blocks_count = 0

        for i in range(len(table_data["text"])):
            if table_data["text"][i].strip() != "":
                if last_block_num != table_data["block_num"][i]:
                    structured_text += "\n"
                    last_block_num = table_data["block_num"][i]
                    text_blocks_count += 1
                structured_text += table_data["text"][i] + " "

        logger.info(
            f"Structured table processing complete with {text_blocks_count} text blocks"
        )
        result = f"Raw Extracted Text:\n{extracted_text}\n\n{structured_text}"
        logger.success("OCR extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during structured data extraction: {e!s}")
        return f"Error while extracting text: {e}"


extract_table_text_remote = MODAL_TOOL_REGISTRY["extract_table_text_modal"]


@tool
def extract_table_text(image_path: str, lang: str = "eng") -> str:
    """
    Extract text from images containing tables using OCR (pyTesseract).
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path (str): Path to the image file containing a table
        lang (str, optional): Language code for OCR. Default: 'eng' for English

    Returns:
        str: String containing the extracted text from the table image
    """
    from PIL import Image

    with Image.open(image_path) as img:
        import io

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        img_bytes = img_byte_arr.getvalue()

    return extract_table_text_remote.remote(img_bytes, lang)


# https://github.com/Kohulan/DECIMER-Image_Transformer
@tool
def decimer_molecule_extraction(image_path: str) -> str:
    """Extract molecule smiles from an image using Decimer.
    Decimer is an open-source tool for Optical Chemical Structure Recognition.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path (str): Path to the image file containing a molecule

    Returns:
        str: String containing the extracted molecule information
    """
    decimer_remote = modal.Function.from_name(
        "rxnenv", "molecule_image_extraction_decimer"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return decimer_remote.remote(image_bytes)


# https://github.com/thomas0809/MolScribe
@tool
def molscribe_molecule_extraction(image_path: str) -> dict[str, Any]:
    """Extract molecule information from an image using MolScribe.
    MolScribe is an image-to-graph model that translates a molecular image to its chemical structure
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer. Example of the output:
    {
        'smiles': 'Fc1ccc(-c2cc(-c3ccccc3)n(-c3ccccc3)c2)cc1',
        'confidence': 0.9175,
        'atoms': [{'atom_symbol': '[Ph]', 'x': 0.5714, 'y': 0.9523, 'confidence': 0.9127}, ... ],
        'bonds': [{'bond_type': 'single', 'endpoint_atoms': [0, 1], 'confidence': 0.9999}, ... ]
    }

    Args:
        image_path (str): Path to the image file containing a molecule

    Returns:
        str: String containing the extracted molecule information
    """
    molscribe_remote = modal.Function.from_name(
        "rxnenv", "molecule_image_extraction_molscribe"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return molscribe_remote.remote(image_bytes)


# https://github.com/thomas0809/MolScribe
@tool
def rxnscribe_reaction_extraction(image_path: str) -> list[dict]:
    """Extract reaction information from an image using RXNScribe.
    RxnScribe is a sequence generation model for reaction diagram parsing.
    This is, from an image it returns:
    {  # First reaction
        'reactants': [
            {
                'category': '[Mol]', 'category_id': 1, 'bbox': (0.1550, 0.0246, 0.2851, 0.2614),
                'smiles': '*OC(=O)c1ccccc1C#Cc1ccccc1',
            },
            # ... more reactants
        ],
        'conditions': [
            {
                'category': '[Txt]', 'category_id': 2, 'bbox': (0.2941, 0.0641, 0.3811, 0.1450),
                'text': ['CIBcat', '(1.4 equiv)']
            },
            # ... more conditions
        ],
        'products': [
            # ...
        ]
    },
    # More reactions

    Args:
        image_path (str): Path to the image file containing a reaction

    Returns:
        list[dict]: List of dictionaries containing the extracted reaction information
    """
    rxnscribe_remote = modal.Function.from_name("rxnenv", "rxn_schema_extraction")
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return rxnscribe_remote.remote(image_bytes)


@tool
def crop_plot_with_labels(image_path: str, output_path: str) -> str:
    """
    Extracts a plot from an image while preserving the axis labels and saves it to the specified path.
    Perfect when the plot contains noise as text or other elements
    around which can make difficult to extract the information from
    the plot to specialized tools.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path where the extracted plot will be saved

    Returns:
        str: Confirmation message indicating where the cropped image was saved
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use binary thresholding to identify the plot background (usually white)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the inverted threshold image
    contours, _ = cv2.findContours(~thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback to Canny edge detection if thresholding doesn't work well
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    # Find the largest contour (assuming it's the plot area)
    if contours:
        plot_contour = max(contours, key=cv2.contourArea)

        # Get bounding box of the plot
        x, y, w, h = cv2.boundingRect(plot_contour)

        # Automatically determine the padding needed for left (y-axis) and bottom (x-axis)
        # by analyzing text regions around the plot

        # Analyze left side for y-axis labels and title
        search_width = min(150, x)  # Use up to 150px or available space
        left_region = gray[
            max(0, y - 30) : min(y + h + 30, image.shape[0]),
            max(0, x - search_width) : x,
        ]

        if left_region.size > 0:
            # Use text detection with more aggressive threshold to catch all text
            left_thresh = cv2.threshold(left_region, 200, 255, cv2.THRESH_BINARY_INV)[1]
            left_contours, _ = cv2.findContours(
                left_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if left_contours:
                # Find the leftmost text element
                if len(left_contours) > 0:
                    leftmost_x = min([cv2.boundingRect(c)[0] for c in left_contours])
                    padding_left = max(search_width - leftmost_x, 0)
                    # Add a bit extra to ensure we catch all of the text
                    padding_left += 10
                else:
                    padding_left = int(w * 0.15)  # More generous default
            else:
                padding_left = int(w * 0.15)
        else:
            padding_left = int(w * 0.15)

        # Analyze bottom side for x-axis labels
        search_height = min(120, image.shape[0] - (y + h))
        bottom_region = gray[
            y + h : min(y + h + search_height, image.shape[0]),
            max(0, x - 20) : min(x + w + 20, image.shape[1]),
        ]

        if bottom_region.size > 0:
            # Use text detection with more aggressive threshold
            bottom_thresh = cv2.threshold(
                bottom_region, 200, 255, cv2.THRESH_BINARY_INV
            )[1]
            bottom_contours, _ = cv2.findContours(
                bottom_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if bottom_contours:
                # Find the lowest text element
                bottom_bounds = [cv2.boundingRect(c) for c in bottom_contours]
                lowest_y = max([b[1] + b[3] for b in bottom_bounds])
                padding_bottom = lowest_y + 10  # Add a small margin
            else:
                padding_bottom = int(h * 0.12)
        else:
            padding_bottom = int(h * 0.12)

        # Use minimal padding for right and top (aggressive cropping)
        padding_right = 1  # Very minimal right padding, extremely aggressive
        padding_top = 1  # Very minimal top padding, extremely aggressive

        # Calculate new coordinates with padding (ensuring we stay within image bounds)
        x1 = max(0, x - padding_left)
        y1 = max(0, y - padding_top)
        x2 = min(image.shape[1], x + w + padding_right)
        y2 = min(image.shape[0], y + h + padding_bottom)

        # Crop the image with padding to include labels
        extracted_plot = image[y1:y2, x1:x2]
    else:
        # If no contours found, use the original image
        extracted_plot = image

    # Save the image to the output path
    cv2.imwrite(output_path, extracted_plot)

    # Return confirmation message
    return f"Cropped image saved in {output_path}"
