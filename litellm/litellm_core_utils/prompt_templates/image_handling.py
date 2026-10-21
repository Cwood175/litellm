"""
Helper functions to handle images passed in messages
"""

import base64

from httpx import Response

import litellm
from litellm import verbose_logger
from litellm.caching.caching import InMemoryCache

MAX_IMGS_IN_MEMORY = 10

in_memory_cache = InMemoryCache(max_size_in_memory=MAX_IMGS_IN_MEMORY)


def _process_image_response(response: Response, url: str) -> str:
    try:
        # If the response object provides raise_for_status, use it so a specific HTTP error is raised
        if hasattr(response, "status_code") and response.status_code != 200:
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            else:
                raise RuntimeError(
                    f"Error: Unable to fetch image from URL. Status code: {getattr(response, 'status_code', None)}, url={url}"
                )
    except Exception:
        # Re-raise so callers and retry logic can handle the specific exception
        raise

    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    image_type = response.headers.get("Content-Type")
    if image_type is None:
        img_type = url.split(".")[-1].lower()
        _img_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(img_type)
        if _img_type is None:
            raise ValueError(
                f"Error: Unsupported image format. Format={img_type}. Supported types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']"
            )
        img_type = _img_type
    else:
        img_type = image_type

    result = f"data:{img_type};base64,{base64_image}"
    in_memory_cache.set_cache(url, result)
    return result


async def async_convert_url_to_base64(url: str) -> str:
    cached_result = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client = litellm.module_level_aclient
    for _ in range(3):
        try:
            response = await client.get(url, follow_redirects=True)
            return _process_image_response(response, url)
        except Exception:
            pass
    raise Exception(f"Error: Unable to fetch image from URL after 3 attempts. url={url}")


def convert_url_to_base64(url: str) -> str:
    cached_result = in_memory_cache.get_cache(url)
    if cached_result:
        return cached_result

    client = litellm.module_level_client
    import asyncio

    for _ in range(3):
        try:
            resp = client.get(url, follow_redirects=True)
            # If client.get returned a coroutine (async client), run it to obtain the response
            if asyncio.iscoroutine(resp):
                response = asyncio.run(resp)
            else:
                response = resp
            return _process_image_response(response, url)
        except Exception as e:
            verbose_logger.exception(e)
            # print(e)
            pass
    raise Exception(f"Error: Unable to fetch image from URL after 3 attempts. url={url}")
