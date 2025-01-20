## Basic example task of writing input file for lammps


Install environment
```bash
cd tasks/lammps_input
uv venv --python 3.12
uv sync
```


Run the environment
```bash
cd tasks/lammps_input/lammps_input
python -m env
```

### Setting Up the API keys

Some of the tools used in this environemtn require API keys.
- `brave_search` needs BraveSearch API key

#### How to set up the API keys
- Create a `.env` file in your working directory
- Add the required API keys in the `.env` file in the following format:
```bash
BRAVE_SEARCH_API_KEY=your_brave_search_api_key
```
- load the keys in the tool module correctly
```python
from dotenv import load_dotenv

load_dotenv("../.env")


@tool
def brave_search(query: str) -> list[dict]:
    """Perform a web search using Brave Search API.

    Args:
        query: The search query string
    Returns:
        A list of dictionaries containing search results with titles and snippets
    Raises:
        ValueError: If BRAVE_SEARCH_API_KEY environment variable is not set
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY environment variable is required but not set"
        )

    brave_search_tool = BraveSearch.from_api_key(api_key=api_key)

    return brave_search_tool._run(query)
```

Alternatively, you can set the keys directly in your environment

- Export the keys directly in your environment
```bash
export BRAVE_SEARCH_API_KEY=your_brave_search_api_key
```
