from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal
import numpy as np
from langchain_community.tools.brave_search.tool import BraveSearch
from modal import Image, Volume
from NSFopen.read import read
from promptstore import PromptStore
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from corral.agents.utils import (
    LiteLLMMessage,
    llm_call,
)
from corral.utils import (
    MODAL_TOOL_REGISTRY,
    embed_text,
    modal_tool,
    tool,
    vector_database_search,
)

store = PromptStore("./prompts")
app = modal.App("macbench")
hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)


@tool
def enhanced_brave_search(
    query: str, num_results: int = 5, min_similarity: float = 0.75
) -> list[dict]:
    """Perform a web search using Brave Search, then filter and rank results using embeddings.

    Args:
        query: The search query string
        num_results: Maximum number of results to return. Defaults to 5
        min_similarity: Minimum similarity score threshold. Defaults to 0.75

    Returns:
        A list of dictionaries containing the most relevant search results
        with their content and metadata, sorted by similarity score

    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable is required but not set"
        )

    try:
        brave_search_tool = BraveSearch.from_api_key(api_key=api_key)
        initial_results = brave_search_tool._run(query)

        if not initial_results:
            return []

        query_embedding = np.array(embed_text(chunks=[query])[0]).reshape(1, -1)
        result_texts = [f"{r['title']}: {r['snippet']}" for r in initial_results]
        result_embeddings = np.array(embed_text(chunks=result_texts))

        similarity_scores = sklearn_cosine_similarity(
            query_embedding, result_embeddings
        ).flatten()

        results_with_scores = []
        for i, result in enumerate(initial_results):
            similarity = float(similarity_scores[i])

            results_with_scores.append(
                {
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "link": result["link"],
                    "similarity_score": similarity,
                }
            )

        filtered_results = [
            r for r in results_with_scores if r["similarity_score"] >= min_similarity
        ]
        sorted_results = sorted(
            filtered_results, key=lambda x: x["similarity_score"], reverse=True
        )

        return sorted_results[:num_results]

    except Exception:
        return []


@tool
def search_lab_safety(query: str) -> list[dict]:
    """Search lab safety knowledge database for relevant information or guidelines.

    Args:
        query: The search query related to laboratory safety

    Returns:
        A list of lab safety information entries with content and relevance scores

    Raises:
        RuntimeError: If the lab safety collection doesn't exist
    """
    return vector_database_search(query, collection_name="lab_safety_collection")


@tool
def search_ms_guide(query: str) -> list[dict]:
    """Search mass spectrometry knowledge database for relevant information or guidelines.

    Args:
        query: The search query related to mass spectrometry

    Returns:
        A list of mass spectrometry information entries with content and relevance scores

    Raises:
        RuntimeError: If the mass spectrometry collection doesn't exist
    """
    return vector_database_search(query, collection_name="ms_guide_collection")


@tool
def search_nmr_guide(query: str) -> list[dict]:
    """Search NMR spectroscopy knowledge database for relevant information or guidelines.

    Args:
        query: The search query related to NMR spectroscopy

    Returns:
        A list of NMR spectroscopy information entries with content and relevance scores

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
        query: The question or instruction related to the image
        image_path: The path to the image file

    Returns:
        The response from the LLM regarding the image analysis
    """

    # Ideally we would like to use the latest model
    model = "gpt-4o-2024-11-20"

    try:
        with Path(image_path).open("rb") as f:
            image_bytes = f.read()

        system_prompt = store.get("9c471e7f-7bbd-4ef4-8068-bf66e838e590")
        system_prompt = system_prompt.fill({})
        user_prompt = [
            {
                "type": "image",
                "content": image_bytes,
            },
            {
                "type": "text",
                "content": f"Query: {query}",
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
    image=Image.debian_slim().pip_install("transformers", "Pillow"),
    gpu="A100-40GB",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def deplot_image_extractor_modal(image_bytes: bytes) -> str:
    """
    Extract data from charts and plots using Google's Deplot model on Modal servers.

    Args:
        image_bytes: Image file data as bytes

    Returns:
        String containing the extracted data table representation
    """
    import io

    from PIL import Image
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    try:
        processor = Pix2StructProcessor.from_pretrained("google/deplot")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

        image = Image.open(io.BytesIO(image_bytes))

        inputs = processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        )
        predictions = model.generate(**inputs)

        return str(processor.decode(predictions[0], skip_special_tokens=True))
    except Exception as e:
        return f"Error while extracting data: {e}"


deplot_image_extractor_remote = MODAL_TOOL_REGISTRY["deplot_image_extractor_modal"]


@tool
def deplot_image_extractor(image_path: str) -> str:
    """
    Extract data from charts and plots using Google's Deplot model.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path: Path to the image file containing a chart or plot

    Returns:
        String containing the extracted data table representation
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
    image=Image.debian_slim().pip_install("transformers", "torch", "Pillow"),
    gpu="A100-40GB",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    scaledown_window=1200,
)
def chart_vllm_bytes(query: str, image_bytes: bytes) -> str:
    """
    Analyze charts and plots using the ChartVLM model.

    Args:
        query: The question or instruction about the chart image
        image_bytes: Image file data as bytes

    Returns:
        String containing the analysis or answer about the chart
    """
    import io

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_name = "U4R/ChartVLM-large"

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name)

        image = Image.open(io.BytesIO(image_bytes))

        inputs = processor(images=image, text=query, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=512, num_beams=3, early_stopping=True
            )

        return str(processor.decode(outputs[0], skip_special_tokens=True))
    except Exception as e:
        return f"Error while extracting data: {e}"


chart_vllm_bytes_remote = MODAL_TOOL_REGISTRY["chart_vllm_bytes"]


@tool
def chart_vllm_extractor(query: str, image_path: str) -> str:
    """
    Analyze charts and plots using the ChartVLM model.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        query: The question or instruction about the chart image
        image_path: Path to the image file containing a chart or plot

    Returns:
        String containing the analysis or answer about the chart
    """
    import io

    from PIL import Image

    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        img_bytes = img_byte_arr.getvalue()

    return chart_vllm_bytes_remote.remote(query, img_bytes)


@modal_tool(
    app=app,
    image=Image.debian_slim()
    .apt_install("tesseract-ocr")
    .pip_install("pytesseract", "opencv-python", "Pillow", "numpy"),
    gpu="A10G",
    memory=1024,
)
def extract_table_text_modal(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from images containing tables using OCR (pyTesseract).

    Args:
        image_bytes: Image file data as bytes
        lang: Language code for OCR. Default: 'eng' for English

    Returns:
        String containing the extracted text from the table image
    """
    import io

    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for OCR
    # Convert to grayscale and apply thresholding
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    custom_config = r"--oem 3 --psm 6 -l " + lang
    extracted_text = pytesseract.image_to_string(processed, config=custom_config)

    try:
        table_data = pytesseract.image_to_data(
            processed, config=custom_config, output_type=pytesseract.Output.DICT
        )
        structured_text = "Structured Table Data:\n"
        last_block_num = -1
        for i in range(len(table_data["text"])):
            if table_data["text"][i].strip() != "":
                if last_block_num != table_data["block_num"][i]:
                    structured_text += "\n"
                    last_block_num = table_data["block_num"][i]
                structured_text += table_data["text"][i] + " "

        return f"Raw Extracted Text:\n{extracted_text}\n\n{structured_text}"
    except Exception as e:
        return f"Error while extracting text: {e}"


extract_table_text_remote = MODAL_TOOL_REGISTRY["extract_table_text_modal"]


@tool
def extract_table_text(image_path: str, lang: str = "eng") -> str:
    """
    Extract text from images containing tables using OCR (pyTesseract).
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path: Path to the image file containing a table
        lang: Language code for OCR. Default: 'eng' for English

    Returns:
        String containing the extracted text from the table image
    """
    from PIL import Image

    with Image.open(image_path) as img:
        import io

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        img_bytes = img_byte_arr.getvalue()

    return extract_table_text_remote.remote(img_bytes, lang)


@tool
def decimer_molecule_extraction(image_path: str) -> str:
    """Extract molecule information from an image using Decimer.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path: Path to the image file containing a molecule

    Returns:
        String containing the extracted molecule information
    """
    decimer_remote = modal.Function.from_name(
        "rxnenv", "molecule_image_extraction_decimer"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return decimer_remote.remote(image_bytes)


@tool
def molscribe_molecule_extraction(image_path: str) -> str:
    """Extract molecule information from an image using MolScribe.
    Ideally the response of this tool should be used compared to the
    response of other tools to determine the best answer.

    Args:
        image_path: Path to the image file containing a molecule

    Returns:
        String containing the extracted molecule information
    """
    molscribe_remote = modal.Function.from_name(
        "rxnenv", "molecule_image_extraction_molscribe"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return molscribe_remote.remote(image_bytes)


@tool
def rxnscribe_reaction_extraction(image_path: str) -> list[dict]:
    """Extract reaction information from an image using RXNScribe.

    Args:
        image_path: Path to the image file containing a reaction

    Returns:
        List of dictionaries containing the extracted reaction information
    """
    rxnscribe_remote = modal.Function.from_name("rxnenv", "rxn_schema_extraction")
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return rxnscribe_remote.remote(image_bytes)


# Tool from AILA (arXiv:2501.10385)
@tool
def afm_image_analyzer(
    image_path: str,
    calculate_friction: bool = False,
    calculate_mean_roughness: bool = False,
    calculate_rms_roughness: bool = False,
) -> dict[str, str | Any]:
    """
    Display and return the image data from the given path of an AFM image.

    Additionally, calculate the following if requested:
    - Average Friction
    - Mean Roughness
    - RMS Roughness

    Args:
        image_path: Path to the image file.
        calculate_friction: Whether to calculate average friction. Defaults to False.
        calculate_mean_roughness: Whether to calculate mean roughness. Defaults to False.
        calculate_rms_roughness: Whether to calculate RMS roughness. Defaults to False.

    Returns:
    - dict[str, str | Any]: A dictionary containing the status, image data, or an error message.
    """
    try:
        # Read the file
        afm = read(image_path)

        # Extract data and parameters
        data = afm.data  # Raw data

        # Assuming 'Image', 'Forward', and 'Z-Axis' are keys in the data structure
        image_data = data["Image"]["Forward"]["Z-Axis"]

        # Calculate Average Friction if requested
        if calculate_friction:
            friction = 0.5 * (
                data["Image"]["Forward"]["Friction force"]
                - data["Image"]["Backward"]["Friction force"]
            )
            average_friction = np.mean(friction)

        # Calculate Mean Roughness if requested
        if calculate_mean_roughness:
            z = data["Image"]["Forward"]["Z-Axis"]
            z_mean = np.mean(z)
            absolute_differences = np.abs(z - z_mean)
            total_sum = np.sum(absolute_differences)
            M, N = z.shape
            mean_roughness = total_sum / (M * N)

        # Calculate RMS Roughness if requested
        if calculate_rms_roughness:
            z = data["Image"]["Forward"]["Z-Axis"]
            z_mean = np.mean(z)
            squared_differences = (z - z_mean) ** 2
            total_sum = np.sum(squared_differences)
            M, N = z.shape
            rms_roughness = np.sqrt(total_sum / (M * N))

        # Return the image data along with status
        result = {
            "status": "Success",
            "message": f"Raw Image {image_path} processed successfully.",
            "image_data": image_data,
        }

        # Include calculated metrics in the result if they were calculated
        if calculate_friction:
            result["average_friction"] = average_friction
        if calculate_mean_roughness:
            result["mean_roughness"] = mean_roughness
        if calculate_rms_roughness:
            result["rms_roughness"] = rms_roughness

        return result

    except Exception as e:
        return {"status": "Error", "message": f"An error occurred: {e!s}"}
