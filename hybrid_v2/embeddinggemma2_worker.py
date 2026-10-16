"""JSON-lines worker. Framework chatter is redirected away from the protocol."""
from __future__ import annotations

import contextlib
import json
import socket
import sys

from .embeddinggemma2 import EmbeddingGemma2Encoder, EmbeddingGemma2Error


def main():
    def blocked(*args, **kwargs):
        raise RuntimeError("EG2_EXTERNAL_NETWORK_PROHIBITED")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    encoder = None
    for line in sys.stdin:
        request = json.loads(line)
        response = {"id": request["id"]}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                operation = request["operation"]
                if operation == "load":
                    if encoder is None:
                        encoder = EmbeddingGemma2Encoder(request["config"]).load()
                    result = encoder.metrics
                elif operation in ("text", "video") and encoder is not None:
                    vector, metrics = (encoder.encode_text(request["query"]) if operation == "text"
                                       else encoder.encode_video(request["path"], request["config"]))
                    result = {"embedding": vector.tolist(), "metrics": metrics}
                else:
                    raise EmbeddingGemma2Error("EG2_INVALID_WORKER_OPERATION")
            response["result"] = result
        except Exception as exc:
            response["error"] = {"code": getattr(exc, "code", "EG2_WORKER_FAILED"), "detail": repr(exc)}
        print(json.dumps(response), flush=True)
    if encoder:
        encoder.close()


if __name__ == "__main__":
    main()
