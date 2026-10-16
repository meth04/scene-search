"""Small isolated HTTP worker: byte cap before JSON decoding; parent enforces deadline."""
import json
import sys


def request(data):
    import httpx

    raw = bytearray()
    try:
        with httpx.Client(timeout=data["timeout"], trust_env=False) as client:
            with client.stream("POST", data["url"], json=data["payload"], headers=data["headers"]) as response:
                status = response.status_code
                for chunk in response.iter_bytes(chunk_size=4096):
                    raw.extend(chunk)
                    if len(raw) > data["byte_cap"]:
                        return {"error": "QWEN_RESPONSE_BYTE_CAP", "http_status": status,
                                "response_bytes": len(raw), "exception_class": "ResponseByteCap"}
        return {"body": json.loads(raw), "http_status": status, "response_bytes": len(raw)}
    except httpx.TimeoutException as exc:
        return {"error": "QWEN_TIMEOUT", "response_bytes": len(raw), "exception_class": type(exc).__name__}
    except httpx.TransportError as exc:
        return {"error": "QWEN_TRANSPORT", "response_bytes": len(raw), "exception_class": type(exc).__name__}
    except ValueError as exc:
        return {"error": "QWEN_SCHEMA_FAILURE", "response_bytes": len(raw), "exception_class": type(exc).__name__}


if __name__ == "__main__":
    print(json.dumps(request(json.load(sys.stdin))))
