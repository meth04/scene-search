from __future__ import annotations

from fractions import Fraction

from src.indexing.domain.windows import SourceFrameRef, source_uuid
from src.indexing.video.window_sampling import nearest_samples, pts_ms


class PTSDecoder:
    """Streaming nearest-frame sampler; keeps at most 24 RGB frames per window."""
    def sample(self, path, asset, window):
        import av

        candidates = []
        # Keep only the two closest frames for each target instead of every decoded image.
        best = [[] for _ in range(12)]
        with av.open(str(path)) as container:
            stream = container.streams[asset.stream_index]
            base = Fraction(asset.time_base_num, asset.time_base_den)
            seek_pts = asset.timeline_origin_pts + int(Fraction(str(window.start_ms))/1000/base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            last = -1.0
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("SOURCE_INVALID: decoded frame has no PTS")
                timestamp = pts_ms(frame.pts, asset.timeline_origin_pts, base.numerator, base.denominator)
                if timestamp < window.start_ms:
                    continue
                if timestamp >= window.end_ms:
                    break
                if timestamp < last:
                    raise ValueError("SOURCE_INVALID: decoded PTS decreased")
                last = timestamp
                ref = SourceFrameRef(
                    frame_id=source_uuid("frame", asset.asset_id, stream.index, frame.pts, base.numerator, base.denominator),
                    asset_id=asset.asset_id, stream_index=stream.index, pts=frame.pts,
                    time_base_num=base.numerator, time_base_den=base.denominator, timestamp_ms=timestamp,
                )
                image = None
                for i, target in enumerate(window.targets):
                    if len(best[i]) < 2 or (abs(timestamp-target), timestamp) < (abs(best[i][-1][0].timestamp_ms-target), best[i][-1][0].timestamp_ms):
                        if image is None:
                            image = frame.to_ndarray(format="rgb24")
                        best[i].append((ref, image))
                        best[i].sort(key=lambda f: (abs(f[0].timestamp_ms-target), f[0].timestamp_ms))
                        best[i] = best[i][:2]
            for items in best:
                candidates.extend(items)
        return nearest_samples(candidates, window)
