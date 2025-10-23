import json
from typing import Any

import anthropic
import tiktoken
from litellm import model_cost

client = anthropic.Anthropic()


def get_context_window(model: str) -> int:
    """
    Return the total context window (max tokens) for a given LiteLLM model name.

    Tries litellm.get_max_tokens(model) first, then falls back to the
    model cost/context map. Returns None if the model isn't known.

    Args:
        model (str): The model name, e.g., "gpt-4o", "claude-haiku-4-5".

    Returns:
        int: The max input tokens for the model, or None if unknown.
    """
    return model_cost.get(model, {}).get("max_input_tokens", None)


def count_anthropic_tokens(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    """
    Count tokens for an Anthropic model request.
    This function ONLY counts input-side tokens (system + tools + messages).

    Args:
        model (str): e.g., "claude-2"
        messages (list[dict[str, Any]]): list of message objects.
        tools (list[dict[str, Any]]): list of tool definitions.
    """
    # Convert OpenAI-style tools to Anthropic format
    corrected_tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {}),
        }
        for tool in (tools or [])
    ]

    if messages[0].get("role") == "system":
        system = str(messages[0]["content"])
        messages = messages[1:]
    else:
        system = ""

    # Messages are already stringified by count_tokens_and_add before reaching here
    try:
        # The SDK handles proper serialization and token accounting internally.
        resp = client.messages.count_tokens(
            model=model,
            messages=messages,
            system=system,
            tools=corrected_tools if corrected_tools else None,
        )
        if isinstance(resp, dict) and "input_tokens" in resp:
            return int(resp["input_tokens"])
        # Some SDK versions return a dataclass-like object
        input_tokens = getattr(resp, "input_tokens", None)
        if input_tokens is not None:
            return int(input_tokens)
    except Exception as e:
        raise ValueError("Failed to count tokens using Anthropic SDK: " + str(e)) from e

    return 100000


def count_openai_tokens(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    """
    Estimate the number of tokens in an OpenAI chat request for a given `model`,
    counting BOTH the `messages` and (optionally) the `tools` specification.

    This function:
      - Handles roles: "system", "user", "assistant", and "tool".
      - Accounts for assistant tool calls (the `tool_calls` field).
      - Works whether message `content` is a string or a rich list (e.g., text/ image parts).
      - Counts `tools` as the third argument (after `model`, `messages`) when provided.

    Args:
        model: Model name string (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo-0125").
        messages: OpenAI Chat Completions messages payload (list of dicts).
        tools: Optional OpenAI "tools" schema (list of dicts).
        return_breakdown: If True, returns a dict with per-section counts instead of a single int.

    Returns:
        Either an int (total tokens) or a dict:
            {
              "total": <int>,
              "messages": <int>,
              "tools": <int>
            }

    Notes & caveats:
        • This uses tokenizer-based estimation. Exact accounting of ChatML/system tokens and
          internal message framing may differ slightly from the service's billing.
        • If `tiktoken` is unavailable, falls back to ~bytes/4 heuristic.
        • Newer OpenAI models commonly use the "o200k_base" encoding; older models often use
          "cl100k_base". We pick a best-effort encoding automatically.
    """
    enc = _get_encoder_for_model(model)

    # Normalize and serialize messages (only meaningful fields)
    norm_messages = [_normalize_message(m) for m in messages]
    messages_json = json.dumps(norm_messages, separators=(",", ":"), ensure_ascii=False)
    msg_tokens = _encode_len(enc, messages_json)

    # Normalize and serialize tools (if any)
    tools_tokens = 0
    if tools:
        norm_tools = [_normalize_tool(t) for t in tools]
        tools_json = json.dumps(norm_tools, separators=(",", ":"), ensure_ascii=False)
        tools_tokens = _encode_len(enc, tools_json)

    return msg_tokens + tools_tokens


def _encode_len(enc, text: str) -> int:
    return len(enc.encode(text))


def _get_encoder_for_model(model: str):
    """
    Try to get a tokenizer via `tiktoken`. When unknown, choose a sane base encoding.
    Returns an object that has `.encode(str) -> List[int]`, or None if tiktoken isn't available.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")


def _normalize_message(m: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only fields that influence content tokenization.
    Handles:
      - string content
      - list content (multi-part: text, image_url, etc.)
      - assistant tool calls
      - tool role (with tool_call_id + content)
    """
    out: dict[str, Any] = {}

    # role is required
    role = m.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        role = str(role) if role else "user"
    out["role"] = role

    # name is optional but may affect tokens
    if "name" in m and m["name"] is not None:
        out["name"] = m["name"]

    # content can be str or list[dict]
    if "content" in m:
        out["content"] = _normalize_content(m["content"])

    # tool role specifics
    # tool messages usually have tool_call_id + content
    if role == "tool" and "tool_call_id" in m and m["tool_call_id"] is not None:
        out["tool_call_id"] = m["tool_call_id"]

    # assistant tool calls (array of {id, type, function:{name, arguments}})
    if m.get("tool_calls"):
        out["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]

    # function_call (legacy) if present
    if m.get("function_call"):
        # Keep name + arguments as text to be conservative
        fc = m["function_call"]
        out["function_call"] = {
            "name": fc.get("name"),
            "arguments": fc.get("arguments"),
        }

    return out


def _normalize_content(content: Any) -> Any:
    """
    Pass through strings. For lists, keep meaningful fields only.
    Example parts might look like:
      {"type": "text", "text": "..."}
      {"type": "image_url", "image_url": {"url": "...", "detail": "low"}}
      {"type": "input_text", "text": "..."}
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        norm_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            keep = {"type": ptype}
            # keep common fields that carry user-visible tokens
            if "text" in part and part["text"] is not None:
                keep["text"] = part["text"]
            if "image_url" in part and part["image_url"] is not None:
                # include URL + minimal metadata since URLs tokenize
                keep["image_url"] = {
                    k: v for k, v in part["image_url"].items() if k in ("url", "detail")
                }
            # add other possible content-bearing keys conservatively
            for k in ("input_text", "input_audio", "input_image"):
                if k in part and part[k] is not None:
                    keep[k] = part[k]
            norm_parts.append(keep)
        return norm_parts
    # Fallback: JSON-serialize unusual structures to capture their tokens
    try:
        return json.loads(json.dumps(content))
    except Exception:
        return str(content)


def _normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    out = {}
    if "id" in tc and tc["id"] is not None:
        out["id"] = tc["id"]
    # e.g. {"type": "function", "function": {"name": "...", "arguments": "..."}}
    ttype = tc.get("type", "function")
    out["type"] = ttype
    func = tc.get("function")
    if isinstance(func, dict):
        out["function"] = {
            "name": func.get("name"),
            "arguments": func.get("arguments"),
        }
    return out


def _normalize_tool(t: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an OpenAI `tools` item. Today they often look like:
      {"type": "function", "function": {"name": str, "description": str, "parameters": {...}}}
    """
    out = {"type": t.get("type", "function")}
    fn = t.get("function")
    if isinstance(fn, dict):
        keep = {}
        if "name" in fn and fn["name"] is not None:
            keep["name"] = fn["name"]
        if "description" in fn and fn["description"] is not None:
            keep["description"] = fn["description"]
        if "parameters" in fn and fn["parameters"] is not None:
            # include full JSON Schema (can be large, but that's what gets tokenized)
            keep["parameters"] = fn["parameters"]
        out["function"] = keep
    return out


def remove_old_budget_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove any existing CONTEXT_BUDGET messages from the first message in the list.

    Args:
        messages (List[dict[str, Any]]): The messages to filter.

    Returns:
        List[dict[str, Any]]: The filtered messages.
    """
    if messages and "[CONTEXT_BUDGET]" in str(messages[0].get("content", "")):
        # Replace content between CONTEXT_BUDGET tokens with nothing in first message
        content = str(messages[0]["content"])
        start_tag = "[CONTEXT_BUDGET]"
        end_tag = "[/CONTEXT_BUDGET]"

        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            # Keep everything after end_tag (remove budget from beginning)
            messages[0]["content"] = content[end_idx + len(end_tag) :].lstrip()
    return messages


def count_tokens_and_add(
    messages: list[dict[str, Any]], model: str, tools: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Count tokens in messages and add token count as metadata if supported.

    Args:
        messages (list[dict[str, Any]]): The messages to count tokens for.
        model (str): The model to use for token counting.
        tools (dict[str, Any]): The tools available to the agent.

    Returns:
        list[dict[str, Any]]: The messages with token count metadata added.
    """
    # Ensure all message content is properly stringified for Anthropic models
    formatted_messages = []
    for msg in messages:
        formatted_msg = msg.copy()
        if "content" in formatted_msg:
            formatted_msg["content"] = str(formatted_msg["content"])
        if "name" in formatted_msg:
            del formatted_msg["name"]
        formatted_messages.append(formatted_msg)
    messages = formatted_messages

    if "openai" in model or "gpt" in model:
        count = count_openai_tokens(
            model=model.replace("openai/", ""), messages=messages, tools=tools
        )
    elif "anthropic" in model or "claude" in model:
        count = count_anthropic_tokens(
            model=model.replace("anthropic/", ""), messages=messages, tools=tools
        )
    else:
        raise NotImplementedError(f"Token counting not implemented for model: {model}")

    window = get_context_window(model=model)
    if window is None:
        window = get_context_window(model=model.split("/")[-1])

    if window is None:
        window = 8192  # Default to 8k if unknown

    messages = remove_old_budget_message(messages)
    budget_message = f"""[CONTEXT_BUDGET]
model: {model.split("/")[-1]}
max_context_tokens: {window}
prompt_tokens_now: {count}
reserve_for_output: 100
remaining_budget: {window - count - 100}
actions_if_low_budget:
  - avoid reading entire files
  - prefer short answers (<= 150 tokens)
  - summarize or drop thoughts if needed
[/CONTEXT_BUDGET]
"""
    # Ensure content is a string before concatenation
    return budget_message + "\n" + str(messages[0]["content"])
