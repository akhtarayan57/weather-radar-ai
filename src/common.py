import time
import requests


class ApiError(Exception):
    pass


def get_json(url, params=None, headers=None, tries=5):
    """GET with retry on network errors, 429 and 5xx. Raises ApiError with a readable message."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            if i == tries - 1:
                raise ApiError(f"Network error: {e}")
            time.sleep(2 ** i)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise ApiError("API key missing or invalid (check OPENAQ_API_KEY).")
        if r.status_code in (408, 429) or r.status_code >= 500:
            if i == tries - 1:
                raise ApiError(f"API busy or rate limited ({r.status_code}). Try again later.")
            try:
                wait = float(r.headers.get("x-ratelimit-reset") or r.headers.get("Retry-After") or 2 ** i)
            except ValueError:
                wait = 2 ** i
            time.sleep(min(wait, 70) + 1)
            continue
        raise ApiError(f"API error {r.status_code}: {r.text[:200]}")
