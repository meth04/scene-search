class RedisCache:
    def __init__(self, url: str):
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("Install redis to use RedisCache") from exc
        self.client = Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self.client.delete(key)
