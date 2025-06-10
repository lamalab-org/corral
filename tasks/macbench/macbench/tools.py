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
def enhanced_brave_search(query: str, num_results: int = 5) -> list[dict]:
    r"""[BRIEF] Perform a web search using Brave Search, then filter and rank results using embeddings. [\BRIEF]

    [DETAILED] This tool performs a web search using Brave Search API,
    retrieves the top results for a query, and then filters and ranks them based on their
    relevance to the search query using embeddings. It returns a list of the most
    relevant search results, each containing the content and metadata of the
    search result, sorted by similarity score. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when the other specific seach tools are not available or when you need to perform a general web search.
    - When you need to find information that is not available in the local vector database or other specialized databases.
    - When you need to retrieve some information that is not available with the other tools.
    - Recommended for general query seaches that do not require specific databases or structured data. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the other more specific tools are not suitable for the query. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to perform a web search. [\CURRENT]
    3. [FOLLOW_UP] Use the information from the web search combined with retrieved from other tools
    to solve the task validating the tasks with the available tools. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Checks if a BRAVE_SEARCH_API_KEY environment variable is set.
    - If not set, raises a ValueError.
    - Uses the Brave Search API to perform a web search with the provided query.
    - Retrieves the top `num_results` results.
    - Filters and ranks the results based on their relevance to the search query using embeddings.
    - Returns a list of dictionaries containing the most relevant search results,
    each with its content and metadata, sorted by similarity score that is also included. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `online_search("What is the chemical formula of caffeine?", num_results=10)`,
        `online_search("What is the chemical formula of aspirin?", num_results=7)`
        `online_search("What is the boiling point of water?", num_results=5)`,
        `online_search("What is the molecular weight of glucose?", num_results=10)`,
        `online_search("What is the structure of benzene?", num_results=5)`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The search query string [\BRIEF]
                        [DETAILED] The query string to search for in the Brave Search API. It should be a descriptive string that represents the information you are looking for. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What is the chemical formula of caffeine?", "What is the boiling point of water?", "What is the structure of benzene?" [\EXAMPLES]

        num_results (int, optional):
                        [BRIEF] Maximum number of results to return. Defaults to 5 [\BRIEF]
                        [DETAILED] The maximum number of search results to return from the Brave Search API. It should be a positive integer. [\DETAILED]
                        [SYNTACTICAL] Format: "any positive integer (e.g., 5, 10, 20)" [\SYNTACTICAL]
                        [EXAMPLES] Examples: 5, 10, 20 [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant search results with their content and metadata, sorted by similarity score [\BRIEF]
                        [DETAILED] Each dictionary contains the content of the search result, its metadata, and a similarity score indicating how relevant the result is to the search query. The results are sorted by similarity score in descending order. [\DETAILED]
                        [EXAMPLES] Examples: [{"content": "Result 1 content", "metadata": {"source": "some_source.com"}, "similarity_score": 0.95}, {"content": "Result 2 content", "metadata": {"source": "second_source.org"}, "similarity_score": 0.90}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
                        [ERROR_WHEN] If the BRAVE_SEARCH_API_KEY environment variable is not set or if an error occurs during the search. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the BRAVE_SEARCH_API_KEY is not set, or if there is an error in making the API request to Brave Search, such as network issues or invalid query parameters. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid BRAVE_SEARCH_API_KEY environment variable to be set.
        - The number of results returned is limited by the `num_results` parameter, which defaults to 5.
        - The search results are filtered and ranked based on their relevance to the search query using embeddings, which may not always yield the most relevant results.
        - The results content might not be completely accurate.
        - The results content might not be complete.
        - If some error occurs during the search, it will return an empty list.
    [/LIMITATIONS]
    """
    return web_search(
        query=query,
        num_results=num_results,
    )


@tool
def search_lab_safety(query: str) -> list[dict]:
    r"""[BRIEF] Search lab safety knowledge database for relevant information or guidelines. [\BRIEF]

    [DETAILED] This tool searches a specialized database containing lab safety knowledge,
    guidelines, and best practices. It retrieves relevant information based on the provided query,
    returning a list of entries that match the search criteria. Each entry includes the content
    and a relevance score indicating how closely it matches the query. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find specific information or guidelines related to laboratory safety.
    - When you are looking for best practices, safety protocols, or hazard information in a lab environment.
    - Recommended for queries related to lab safety, chemical handling, and emergency procedures. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the task at hand is about lab safety. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to search for lab safety information. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved information to inform lab safety practices, or answer the question at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Searches a specialized lab safety knowledge database using the provided query.
    - Retrieves relevant entries that match the search criteria.
    - Each entry includes the content and a relevance score indicating how closely it matches the query.
    - Returns a list of dictionaries containing the most relevant lab safety information, sorted by relevance score. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_lab_safety("What are the safety protocols for handling flammable chemicals?")`,
        `search_lab_safety("How to handle chemical spills safely?")`,
        `search_lab_safety("What are the emergency procedures for a lab fire?")`,
        `search_lab_safety("What personal protective equipment (PPE) is required in a chemistry lab?")`,
        `search_lab_safety("What are the best practices for working with hazardous materials in a lab?")`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The search query related to laboratory safety [\BRIEF]
                        [DETAILED] The query string to search for in the lab safety knowledge database. It should be a descriptive string that represents the information you are looking for related to lab safety. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What are the safety protocols for handling flammable chemicals?", "How to handle chemical spills safely?", "What are the emergency procedures for a lab fire?" [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant lab safety information with their content and relevance scores, sorted by relevance score [\BRIEF]
                        [DETAILED] Each dictionary contains the content of the lab safety information, its metadata, and a relevance score indicating how relevant the result is to the search query. The results are sorted by relevance score in descending order. [\DETAILED]
                        [EXAMPLES] Examples: [{"content": "Lab safety protocol for flammable chemicals", "metadata": {"source": "safety_manual.pdf"}, "relevance_score": 0.95}, {"content": "Chemical spill handling guidelines", "metadata": {"source": "spill_guide.pdf"}, "relevance_score": 0.90}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If the lab safety collection doesn't exist or if an error occurs during the search. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the lab safety collection is not found in the vector database, or if there is an error in making the search request, such as network issues or invalid query parameters. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the database connection. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The lab safety collection must exist in the vector database for this tool to work.
        - The search results are filtered and ranked based on their relevance to the search query using embeddings, which may not always yield the most relevant results.
        - The results content might not be completely accurate.
        - Since this tool relies on a vector database, it may not return results if the database is not properly indexed or if the query is too vague.
    [/LIMITATIONS]
    """
    return vector_database_search(query, collection_name="lab_safety_collection")


@tool
def search_ms_guide(query: str) -> list[dict]:
    r"""[BRIEF] Search mass spectrometry knowledge database for relevant information or guidelines. [\BRIEF]

    [DETAILED] This tool searches a specialized database containing mass spectrometry knowledge and
    guidelines on how to interpret MS spectra. It retrieves relevant information based on the provided query,
    returning a list of entries that match the search criteria. Each entry includes the content
    and a relevance score indicating how closely it matches the query. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find specific information or guidelines related to mass spectrometry.
    - Recommended for queries related to mass spectrometry, MS spectra interpretation, and mass spectrometry techniques. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the task at hand is about mass spectrometry or MS spectra interpretation. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to search for mass spectrometry information. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved information to inform mass spectrometry analysis, or answer the question at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Searches a specialized mass spectrometry knowledge database using the provided query.
    - Retrieves relevant entries that match the search criteria.
    - Each entry includes the content and a relevance score indicating how closely it matches the query.
    - Returns a list of dictionaries containing the most relevant mass spectrometry information, sorted by relevance score. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_ms_guide("How to identify bromine?")`,
        `search_ms_guide("What is the difference between bromine and clorine in MS?")`,
        `search_ms_guide("How to identify halides using MS spectra?")`,
        `search_ms_guide("What is the role of mass spectrometry in differientating halides?")`,
        `search_ms_guide("How to analyze MS data for molecules with halides?")`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The search query related to mass spectrometry [\BRIEF]
                        [DETAILED] The query string to search for in the mass spectrometry knowledge database. It should be a descriptive string that represents the information you are looking for related to mass spectrometry. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "How to differiantate bromine from clorine?", "How to identify clorine using MS spectra?" [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant mass spectrometry information with their content and relevance scores, sorted by relevance score [\BRIEF]
                        [DETAILED] Each dictionary contains the content of the mass spectrometry information, its metadata, and a relevance score indicating how relevant the result is to the search query. The results are sorted by relevance score in descending order. [\DETAILED]
                        [EXAMPLES] Examples: [{"content": "Mass spectrum interpretation guidelines", "metadata": {"source": "ms_guide.pdf"}, "relevance_score": 0.95}, {"content": "Common ionization techniques in mass spectrometry", "metadata": {"source": "ionization_techniques.pdf"}, "relevance_score": 0.90}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If the mass spectrometry collection doesn't exist or if an error occurs during the search. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the mass spectrometry collection is not found in the vector database, or if there is an error in making the search request, such as network issues or invalid query parameters. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the database connection. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The mass spectrometry collection must exist in the vector database for this tool to work.
        - The search results are filtered and ranked based on their relevance to the search query using embeddings, which may not always yield the most relevant results.
        - The results content might not be completely accurate.
        - Since this tool relies on a vector database, it may not return results if the database is not properly indexed or if the query is too vague.
    [/LIMITATIONS]
    """
    return vector_database_search(query, collection_name="ms_guide_collection")


@tool
def search_nmr_guide(query: str) -> list[dict]:
    r"""[BRIEF] Search NMR spectroscopy knowledge database for relevant information or guidelines. [\BRIEF]

    [DETAILED] This tool searches a specialized database containing NMR spectroscopy knowledge and
    guidelines on how to interpret NMR spectra. It retrieves relevant information based on the provided query,
    returning a list of entries that match the search criteria. Each entry includes the content
    and a relevance score indicating how closely it matches the query. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to find specific information or guidelines related to NMR spectroscopy.
    - Recommended for queries related to NMR spectroscopy, NMR spectra interpretation, and NMR techniques. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the task at hand is about NMR spectroscopy or NMR spectra interpretation. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string to search for NMR spectroscopy information. [\CURRENT]
    3. [FOLLOW_UP] Use the retrieved information to inform NMR spectroscopy analysis, or answer the question at hand. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Searches a specialized NMR spectroscopy knowledge database using the provided query.
    - Retrieves relevant entries that match the search criteria.
    - Each entry includes the content and a relevance score indicating how closely it matches the query.
    - Returns a list of dictionaries containing the most relevant NMR spectroscopy information, sorted by relevance score. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `search_nmr_guide("How to interpret the proton NMR to differentiate benzene derivatives?")`,
        `search_nmr_guide("What are the differences between the positions of substituents in benzene derivatives?")`,
        `search_nmr_guide("How to identify benzene substituents using NMR?")`,
        `search_nmr_guide("What is the role of the position of the subsituent in aromatic chemical shifts?")`,
        `search_nmr_guide("How to analyze NMR data for aromatic systems?")`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The search query related to NMR spectroscopy [\BRIEF]
                        [DETAILED] The query string to search for in the NMR spectroscopy knowledge database. It should be a descriptive string that represents the information you are looking for related to NMR spectroscopy. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "How to interpret the proton NMR to differentiate benzene derivatives?", "What are the differences between the positions of substituents in benzene derivatives?" [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] A list of dictionaries containing the most relevant NMR spectroscopy information with their content and relevance scores, sorted by relevance score [\BRIEF]
                        [DETAILED] Each dictionary contains the content of the NMR spectroscopy information, its metadata, and a relevance score indicating how relevant the result is to the search query. The results are sorted by relevance score in descending order. [\DETAILED]
                        [EXAMPLES] Examples: [{"content": "NMR spectroscopy interpretation guidelines", "metadata": {"source": "nmr_guide.pdf"}, "relevance_score": 0.95}, {"content": "Common chemical shifts in NMR spectroscopy", "metadata": {"source": "chemical_shifts.pdf"}, "relevance_score": 0.90}, ...] [\EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
                        [ERROR_WHEN] If the NMR spectroscopy collection doesn't exist or if an error occurs during the search. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the NMR spectroscopy collection is not found in the vector database, or if there is an error in making the search request, such as network issues or invalid query parameters. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the database connection. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - The NMR spectroscopy collection must exist in the vector database for this tool to work.
        - The search results are filtered and ranked based on their relevance to the search query using embeddings, which may not always yield the most relevant results.
        - The results content might not be completely accurate.
        - Since this tool relies on a vector database, it may not return results if the database is not properly indexed or if the query is too vague.
    [/LIMITATIONS]
    """
    return vector_database_search(query, collection_name="nmr_guide_collection")


@tool
def llm_vision_expert(query: str, image_path: str) -> str:
    r"""[BRIEF] Analyze an image using a vision expert LLM to extract information from charts, plots, and tables. [\BRIEF]

    [DETAILED] This tool uses a vision expert LLM to analyze an image and extract relevant information from charts, plots, and tables.
    It takes a query string and an image file path as input, processes the image, and returns a response from the LLM regarding the image analysis. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract information from images containing charts, plots, or tables.
    - When you want to leverage a vision expert LLM to analyze visual data and provide insights.
    - Recommended for tasks that involve interpreting visual data, such as scientific charts, financial plots, or data tables. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains relevant visual data (charts, plots, tables) that needs analysis. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with a descriptive query string and the path to the image file to perform the analysis. [\CURRENT]
    3. [FOLLOW_UP] Use the response from the LLM to combine it with the response from other tools such as `deplot_image_extractor_modal` or `extract_table_text` depending on the case study. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses a vision expert LLM to analyze the provided image.
    - Encodes the image in base64 format to send it to the LLM.
    - Constructs a system prompt and user prompt to guide the LLM in analyzing the image.
    - Sends the prompts to the LLM and retrieves the response.
    - Returns the response from the LLM regarding the image analysis. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `llm_vision_expert("What does this chart show?", "path/to/chart_image.png")`,
        `llm_vision_expert("Can you extract the data from this table?", "path/to/table_image.png")`,
        `llm_vision_expert("What trends can you identify in this plot?", "path/to/plot_image.png")`,
        `llm_vision_expert("Analyze the data presented in this image.", "path/to/data_image.png")`,
        `llm_vision_expert("What insights can you provide from this scientific chart?", "path/to/scientific_chart.png")`,
    ]
    [\SYNTACTICAL]

    Args:
        query (str):
                        [BRIEF] The question or instruction related to the image [\BRIEF]
                        [DETAILED] The query string that describes what information you want to extract from the image. It should be a descriptive string that represents the analysis you want to perform on the visual data. [\DETAILED]
                        [SYNTACTICAL] Format: "string with no special requirements" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "What does this chart show?", "Can you extract the data from this table?", "What trends can you identify in this plot?" [\EXAMPLES]

        image_path (str):
                        [BRIEF] Path to the image file containing visual data [\BRIEF]
                        [DETAILED] The file path to the image that contains charts, plots, or tables that need analysis. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/chart_image.png", "path/to/table_image.jpg", "path/to/plot_image.jpeg" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The response from the LLM regarding the image analysis [\BRIEF]
                        [DETAILED] A string containing the response from the vision expert LLM, which includes the analysis of the image based on the provided query. The response may include insights, data extraction, or interpretations of the visual data in the image. [\DETAILED]
                        [EXAMPLES] Examples: "The chart shows a significant increase in sales over the last quarter.", "The table contains data on the monthly expenses for the year.", "The plot indicates a positive correlation between the two variables." [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the response from the LLM. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, encoding it in base64, or if there is an error in making the LLM call, such as network issues or invalid prompts. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain relevant visual data (charts, plots, tables) for the analysis to be meaningful.
        - The LLM response may not always be accurate or complete, depending on the complexity of the visual data and the query.
        - The tool relies on the availability of the LLM and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
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
    """[BRIEF] Extract data from charts and plots using Google's Deplot model. [\\BRIEF]

    [DETAILED] This tool extracts data from images containing charts or plots using Google's Deplot model.
    It takes the path to an image file as input, processes the image, and returns a string representation of the extracted data table. [\\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract structured data from images of charts or plots.
    - When you want to convert visual data representations into a machine-readable format.
    - Recommended for tasks that involve analyzing visual data, such as scientific charts, or any kind of data plots. [\\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains relevant visual data (charts, plots) that needs analysis. [\\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the image file to perform the extraction. [\\CURRENT]
    3. [FOLLOW_UP] Use the extracted data to inform further analysis, or answer questions. It is reccomended to combine it with other data sources such as `llm_vision_expert`. [\\FOLLOW_UP] [\\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses Google's Deplot model to analyze the provided image.
    - Reads the image file and converts it to bytes.
    - Sends the image bytes to the Deplot model for processing.
    - Returns a string representation of the extracted data table from the image. [\\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `deplot_image_extractor("path/to/chart_image.png")`,
        `deplot_image_extractor("path/to/plot_image.jpg")`,
        `deplot_image_extractor("path/to/data_table_image.png")`,
        `deplot_image_extractor("path/to/scientific_chart.png")`,
        `deplot_image_extractor("path/to/financial_plot.jpeg")`,
    ]
    [\\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the image file containing a chart or plot [\\BRIEF]
                        [DETAILED] The file path to the image that contains charts or plots from which data needs to be extracted. It should be a valid file path pointing to an image file. [\\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/chart_image.png", "path/to/plot_image.jpg", "path/to/data_table_image.png" [\\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The extracted data table representation from the image [\\BRIEF]
                        [DETAILED] A string containing the extracted data table representation from the image. The data is structured in a way that can be easily interpreted or used for further analysis. [\\DETAILED]
                        [EXAMPLES] Examples: "Column1, Column2, Column3\nValue1, Value2, Value3\n..." [\\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the extracted data. [\\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, converting it to bytes, or if there is an error in making the Deplot model call, such as network issues or invalid image format. [\\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\\ERROR_RECOVERY]
    [\\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain relevant visual data (charts, plots) for the extraction to be meaningful.
        - The Deplot model response may not always be accurate or complete, depending on the complexity of the visual data.
        - The tool relies on the availability of the Deplot model and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
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
    """[BRIEF] Extract text from images containing tables using OCR (pyTesseract). [\\BRIEF]

    [DETAILED] This tool extracts text from images containing tables using Optical Character Recognition (OCR) with pyTesseract.
    It takes the path to an image file as input, processes the image, and returns a string containing the extracted text from the table. [\\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract text data from images of tables.
    - When you want to convert visual table data into a machine-readable text format.
    - Recommended for tasks that involve analyzing tabular data in images, such as scientific tables, or any kind of structured data representation. [\\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains a table from which text needs to be extracted. [\\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the image file and the desired language for OCR to perform the extraction. [\\CURRENT]
    3. [FOLLOW_UP] Use the extracted text to inform further analysis, or answer questions. It is reccomended to combine it with other data sources such as `llm_vision_expert`. [\\FOLLOW_UP] [\\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses pyTesseract to perform OCR on the provided image.
    - Reads the image file and converts it to bytes.
    - Preprocesses the image to enhance text recognition (grayscale conversion, thresholding, morphological operations).
    - Extracts raw text and structured table data from the image.
    - Returns a string containing both the raw extracted text and the structured table data from the image. [\\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `extract_table_text("path/to/table_image.png")`,
        `extract_table_text("path/to/table_image.jpg", lang="eng")`,
        `extract_table_text("path/to/data_table_image.png", lang="fra")`,
        `extract_table_text("path/to/scientific_table.png", lang="spa")`,
        `extract_table_text("path/to/financial_table.jpeg", lang="deu")`,
    ]
    [\\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the image file containing a table [\\BRIEF]
                        [DETAILED] The file path to the image that contains a table from which text needs to be extracted. It should be a valid file path pointing to an image file. [\\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/table_image.png", "path/to/table_image.jpg", "path/to/data_table_image.png" [\\EXAMPLES]

        lang (str, optional):
                        [BRIEF] Language code for OCR (default: 'eng') [\\BRIEF]
                        [DETAILED] The language code for OCR processing. It should be a valid language code supported by Tesseract OCR. Default is 'eng' for English. [\\DETAILED]
                        [SYNTACTICAL] Format: "language code string" [\\SYNTACTICAL]
                        [EXAMPLES] Examples: "eng", "fra", "spa", "deu" [\\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The extracted text from the table image [\\BRIEF]
                        [DETAILED] A string containing the raw extracted text and structured table data from the image. The text is formatted to include both the raw extraction and a structured representation of the table data. [\\DETAILED]
                        [EXAMPLES] Examples: "Raw Extracted Text:\nColumn1, Column2, Column3\nValue1, Value2, Value3\n\nStructured Table Data:\nColumn1: Value1\nColumn2: Value2\n..." [\\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the extracted text. [\\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, converting it to bytes, or if there is an error in making the OCR call, such as network issues or invalid image format. [\\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\\ERROR_RECOVERY]
    [\\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain a table from which text can be extracted.
        - The OCR results may not always be accurate, especially if the image quality is poor or the text is not clearly visible.
        - The tool relies on the availability of Tesseract OCR and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
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
    r"""[BRIEF] Extract molecule smiles from an image using Decimer. [\BRIEF]

    [DETAILED] This tool extracts molecule information from an image using Decimer, an open-source tool for Optical Chemical Structure Recognition.
    It takes the path to an image file as input, processes the image, and returns a string containing the extracted molecule information. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract chemical structure information from images of molecules.
    - When you want to convert visual representations of molecules into a machine-readable format (SMILES).
    - Recommended for tasks that involve analyzing chemical structures in images. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains a clear representation of a molecule that needs to be analyzed. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the image file to perform the extraction. [\CURRENT]
    3. [FOLLOW_UP] Use the extracted molecule information to inform further chemical analysis, or answer questions about the molecule. It is reccomended to combine it with other data sources such as `molscribe_molecule_extraction`. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses Decimer to analyze the provided image.
    - Reads the image file and converts it to bytes.
    - Sends the image bytes to the Decimer model for processing.
    - Returns a string containing the extracted molecule information, typically in SMILES format. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `decimer_molecule_extraction("path/to/molecule_image.png")`,
        `decimer_molecule_extraction("path/to/chemical_structure_image.jpg")`,
        `decimer_molecule_extraction("path/to/molecule_diagram_image.jpeg")`,
        `decimer_molecule_extraction("path/to/organic_compound_image.png")`,
        `decimer_molecule_extraction("path/to/inorganic_structure_image.jpg")`,
    ]
    [\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the image file containing a molecule [\BRIEF]
                        [DETAILED] The file path to the image that contains a molecule from which information needs to be extracted. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/molecule_image.png", "path/to/chemical_structure_image.jpg", "path/to/molecule_diagram_image.jpeg" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] The extracted molecule information from the image [\BRIEF]
                        [DETAILED] A string containing the extracted molecule information, typically in SMILES format. The information is structured in a way that can be easily interpreted or used for further chemical analysis. [\DETAILED]
                        [EXAMPLES] Examples: "C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1C(=O)O" [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the extracted molecule information. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, converting it to bytes, or if there is an error in making the Decimer model call, such as network issues or invalid image format. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain a clear representation of a molecule for the extraction to be meaningful.
        - The Decimer model response may not always be accurate or complete, depending on the complexity of the visual data and the quality of the image.
        - The tool relies on the availability of the Decimer model and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
    """
    decimer_remote = modal.Function.from_name(
        "chemenv", "molecule_image_extraction_decimer"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return decimer_remote.remote(image_bytes)


# https://github.com/thomas0809/MolScribe
@tool
def molscribe_molecule_extraction(image_path: str) -> dict[str, Any]:
    r"""[BRIEF] Extract molecule information from an image using MolScribe. [\BRIEF]

    [DETAILED] This tool extracts molecule information from an image using MolScribe, an image-to-graph model that translates a molecular image to its chemical structure.
    It takes the path to an image file as input, processes the image, and returns a dictionary containing the extracted molecule information, including SMILES representation, confidence scores, atom details, and bond information. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract detailed chemical structure information from images of molecules.
    - When you want to convert visual representations of molecules into a structured format that includes atom and bond details.
    - Recommended for tasks that involve analyzing chemical structures in images, such as molecular diagrams or chemical representations. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains a clear representation of a molecule that needs to be analyzed. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the image file to perform the extraction. [\CURRENT]
    3. [FOLLOW_UP] Use the extracted molecule information to inform further chemical analysis, or answer questions about the molecule. It is reccomended to compare the response of this tool with other tools such as `decimer_molecule_extraction` to determine the best answer. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses MolScribe to analyze the provided image.
    - Reads the image file and converts it to bytes.
    - Sends the image bytes to the MolScribe model for processing.
    - Returns a dictionary containing the extracted molecule information, including SMILES representation, confidence scores, atom details (symbol, coordinates, confidence), and bond information (type, endpoint atoms, confidence). [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `molscribe_molecule_extraction("path/to/molecule_image.png")`,
        `molscribe_molecule_extraction("path/to/chemical_structure_image.jpg")`,
        `molscribe_molecule_extraction("path/to/molecule_diagram_image.jpeg")`,
        `molscribe_molecule_extraction("path/to/organic_compound_image.png")`,
        `molscribe_molecule_extraction("path/to/inorganic_structure_image.jpg")`,
    ]
    [\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the image file containing a molecule [\BRIEF]
                        [DETAILED] The file path to the image that contains a molecule from which information needs to be extracted. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/molecule_image.png", "path/to/chemical_structure_image.jpg", "path/to/molecule_diagram_image.jpeg" [\EXAMPLES]

    Returns:
        dict[str, Any]:
                        [BRIEF] The extracted molecule information from the image [\BRIEF]
                        [DETAILED] A dictionary containing the extracted molecule information, including:
                        - 'smiles': SMILES representation of the molecule
                        - 'confidence': Confidence score of the extraction
                        - 'atoms': List of dictionaries with atom details (symbol, coordinates, confidence)
                        - 'bonds': List of dictionaries with bond details (type, endpoint atoms, confidence) [\DETAILED]
                        [EXAMPLES] Examples: {
                            'smiles': 'Fc1ccc(-c2cc(-c3ccccc3)n(-c3ccccc3)c2)cc1',
                            'confidence': 0.9175,
                            'atoms': [{'atom_symbol': '[Ph]', 'x': 0.5714, 'y': 0.9523, 'confidence': 0.9127}, ... ],
                            'bonds': [{'bond_type': 'single', 'endpoint_atoms': [0, 1], 'confidence': 0.9999}, ... ]
                        } [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the extracted molecule information. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, converting it to bytes, or if there is an error in making the MolScribe model call, such as network issues or invalid image format. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain a clear representation of a molecule for the extraction to be meaningful.
        - The MolScribe model response may not always be accurate or complete, depending on the complexity of the visual data and the quality of the image.
        - The tool relies on the availability of the MolScribe model and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
    """
    molscribe_remote = modal.Function.from_name(
        "chemenv", "molecule_image_extraction_molscribe"
    )
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return molscribe_remote.remote(image_bytes)


# https://github.com/thomas0809/MolScribe
@tool
def rxnscribe_reaction_extraction(image_path: str) -> list[dict]:
    r"""[BRIEF] Extract reaction information from an image using RXNScribe. [\BRIEF]

    [DETAILED] This tool extracts reaction information from an image using RXNScribe, a sequence generation model for reaction diagram parsing.
    It takes the path to an image file as input, processes the image, and returns a list of dictionaries containing the extracted reaction information, including reactants, conditions, and products. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you need to extract chemical reaction information from images of reaction diagrams.
    - When you want to convert visual representations of chemical reactions into a structured format that includes reactants, conditions, and products.
    - Recommended for tasks that involve analyzing chemical reactions in images, such as reaction schemes or chemical process diagrams. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains a clear representation of a chemical reaction that needs to be analyzed. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the image file to perform the extraction. [\CURRENT]
    3. [FOLLOW_UP] Use the extracted reaction information to inform further chemical analysis, or answer questions about the reaction. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses RXNScribe to analyze the provided image.
    - Reads the image file and converts it to bytes.
    - Sends the image bytes to the RXNScribe model for processing.
    - Returns a list of dictionaries containing the extracted reaction information, including reactants, conditions, and products. Each dictionary represents a reaction and contains details such as category, bounding box coordinates, SMILES representation for reactants, and text for conditions. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `rxnscribe_reaction_extraction("path/to/reaction_image.png")`,
        `rxnscribe_reaction_extraction("path/to/chemical_reaction_image.jpg")`,
        `rxnscribe_reaction_extraction("path/to/reaction_diagram_image.jpeg")`,
        `rxnscribe_reaction_extraction("path/to/organic_reaction_image.png")`,
        `rxnscribe_reaction_extraction("path/to/inorganic_reaction_image.jpg")`,
    ]
    [\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the image file containing a chemical reaction [\BRIEF]
                        [DETAILED] The file path to the image that contains a chemical reaction from which information needs to be extracted. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/reaction_image.png", "path/to/chemical_reaction_image.jpg", "path/to/reaction_diagram_image.jpeg" [\EXAMPLES]

    Returns:
        list[dict]:
                        [BRIEF] The extracted reaction information from the image [\BRIEF]
                        [DETAILED] A list of dictionaries containing the extracted reaction information, where each dictionary represents a reaction and includes:
                        - 'reactants': List of reactants with details such as category, bounding box coordinates, and SMILES representation
                        - 'conditions': List of conditions with details such as category, bounding box coordinates, and text
                        - 'products': List of products with similar structure to reactants [\DETAILED]
                        [EXAMPLES] Examples: [
                            {
                                'reactants': [{'category': '[Mol]', 'category_id': 1, 'bbox': (0.1550, 0.0246, 0.2851, 0.2614), 'smiles': '*OC(=O)c1ccccc1C#Cc1ccccc1'}],
                                'conditions': [{'category': '[Txt]', 'category_id': 2, 'bbox': (0.2941, 0.0641, 0.3811, 0.1450), 'text': ['CIBcat', '(1.4 equiv)']}],
                                'products': []
                            },
                            # More reactions
                        ] [\EXAMPLES]

    [RAISES] Exceptions:
        Exception:
                        [ERROR_WHEN] If an error occurs while processing the image or retrieving the extracted reaction information. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when there is an issue with reading the image file, converting it to bytes, or if there is an error in making the RXNScribe model call, such as network issues or invalid image format. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Try a different tool or check the image file path and format. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain a clear representation of a chemical reaction for the extraction to be meaningful.
        - The RXNScribe model response may not always be accurate or complete, depending on the complexity of the visual data and the quality of the image.
        - The tool relies on the availability of the RXNScribe model and its ability to process images, which may have limitations in terms of image size and format.
    [/LIMITATIONS]
    """
    rxnscribe_remote = modal.Function.from_name("chemenv", "rxn_schema_extraction")
    with Path(image_path).open("rb") as f:
        image_bytes = f.read()

    return rxnscribe_remote.remote(image_bytes)


@tool
def crop_plot_with_labels(image_path: str, output_path: str) -> str:
    r"""[BRIEF] Extracts a plot from an image while preserving the axis labels and saves it to the specified path. [\BRIEF]

    [DETAILED] This tool extracts a plot from an image while preserving the axis labels and saves it to the specified output path.
    It is particularly useful when the plot contains noise such as text or other elements around it, which can make it difficult to extract the information from the plot using specialized tools. [\DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use it when you have an image containing a plot with axis labels that you want to extract and save as a separate image.
    - When the plot contains noise or additional elements that make it challenging to extract the plot information using other tools.
    - Recommended for tasks that involve extracting plots from images for further analysis or visualization. [\PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the image contains a plot with axis labels that need to be extracted. Ensure that there are noise around that might make difficult the extraction. [\PREREQUISITE]
    2. [CURRENT] Apply this tool with the path to the input image and the desired output path to perform the extraction. [\CURRENT]
    3. [FOLLOW_UP] Use the extracted plot image for further analysis, visualization, or reporting. [\FOLLOW_UP] [\WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Uses OpenCV to process the provided image.
    - Loads the image and converts it to grayscale.
    - Applies Gaussian blur to reduce noise and binary thresholding to identify the plot background.
    - Finds contours in the inverted threshold image to identify the plot area.
    - Automatically determines the padding needed for the left (y-axis) and bottom (x-axis) by analyzing text regions around the plot.
    - Crops the image with padding to include labels and saves the extracted plot to the specified output path. [\CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `crop_plot_with_labels("path/to/input_image.png", "path/to/output_plot.png")`,
        `crop_plot_with_labels("path/to/plot_image.jpg", "path/to/extracted_plot.jpg")`,
        `crop_plot_with_labels("path/to/data_plot_image.png", "path/to/cropped_plot.png")`,
        `crop_plot_with_labels("path/to/scientific_plot_image.png", "path/to/scientific_cropped_plot.png")`,
        `crop_plot_with_labels("path/to/financial_plot_image.jpeg", "path/to/financial_cropped_plot.jpeg")`,
    ]
    [\SYNTACTICAL]

    Args:
        image_path (str):
                        [BRIEF] Path to the input image containing a plot [\BRIEF]
                        [DETAILED] The file path to the image that contains a plot with axis labels that need to be extracted. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to an image file" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/input_image.png", "path/to/plot_image.jpg", "path/to/data_plot_image.png" [\EXAMPLES]

        output_path (str):
                        [BRIEF] Path where the extracted plot will be saved [\BRIEF]
                        [DETAILED] The file path where the extracted plot image will be saved. It should be a valid file path pointing to an image file. [\DETAILED]
                        [SYNTACTICAL] Format: "valid file path to save the image" [\SYNTACTICAL]
                        [EXAMPLES] Examples: "path/to/output_plot.png", "path/to/extracted_plot.jpg", "path/to/cropped_plot.png" [\EXAMPLES]

    Returns:
        str:
                        [BRIEF] Confirmation message indicating where the cropped image was saved [\BRIEF]
                        [DETAILED] A string containing a confirmation message indicating the path where the cropped image with the extracted plot was saved. This message can be used for further reference or logging. [\DETAILED]
                        [EXAMPLES] Examples: "Cropped plot saved to path/to/output_plot.png", "Extracted plot saved to path/to/extracted_plot.jpg" [\EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError:
                        [ERROR_WHEN] If the input image cannot be loaded or if there is an issue with saving the output image. [\ERROR_WHEN]
                        [ERROR_DETAILS] This exception is raised when the specified input image path does not exist or cannot be read, or if there is an issue with writing the output image to the specified path. [\ERROR_DETAILS]
                        [ERROR_RECOVERY] Ensure that the input image path is correct. [\ERROR_RECOVERY]
    [\RAISES]

    [LIMITATIONS] Known Limitations:
        - Requires a valid image file path to be provided.
        - The image must contain a plot with axis labels for the extraction to be meaningful.
        - The tool relies on OpenCV for image processing, which may have limitations in terms of image size and format. Due to the aggressive cropping, it may not work well with images that have complex backgrounds or multiple plots.
        - The automatic padding detection may not always work perfectly, especially if the plot has unusual layouts or if the labels are not clearly distinguishable from the background.
    [/LIMITATIONS]
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
