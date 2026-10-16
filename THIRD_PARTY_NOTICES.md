# Third-party runtime assets

Scene Search's own source retains the MIT license in `LICENSE`. This does not
relicense third-party models or dependencies. The distribution includes or uses
the following separately licensed components; preserve these notices when sharing it.

| Component | License / attribution | Pinned source |
|---|---|---|
| EmbeddingGemma 2 | Apache-2.0; Google DeepMind | [Model card at the frozen revision](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md) |
| YOLO11n and Ultralytics runtime | AGPL-3.0, or a separate Ultralytics Enterprise license | [Weights](https://github.com/ultralytics/assets/releases/tag/v8.3.0), [runtime source v8.4.171](https://github.com/ultralytics/ultralytics/tree/v8.4.171), [licensing](https://www.ultralytics.com/license) |
| YuNet | MIT; Shiqi Yu | [OpenCV Zoo model and license](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet) |
| YOLOv8 license-plate weights | Upstream model card declares MIT; Koushim. Uses the separately licensed Ultralytics runtime. | [Frozen model revision](https://huggingface.co/Koushim/yolov8-license-plate-detection/tree/9aaa5cd490abe0c165882ba87f4f62658ab54d01) |
| FFmpeg / ffprobe Windows GPL build | GPL and the upstream build's bundled third-party notices | [BtbN build](https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-10-15-13-03), [build scripts](https://github.com/BtbN/FFmpeg-Builds), [FFmpeg source](https://git.ffmpeg.org/ffmpeg.git) |

The public source repository and source archive provide the product source,
installer, build/packaging scripts and pinned dependency manifests:
https://github.com/meth04/scene-search.
Source for dependencies is available at the exact versions referenced above and
in `configs/product/requirements-*.lock`. FFmpeg is downloaded as the original
upstream archive with its build notices, not relabeled as MIT software.
The release also supplies the unmodified corresponding Ultralytics source archive
`ultralytics-8.4.171.tar.gz` (SHA-256
`8583d9d890fb166628f89b9b00bcb2465bc0ce1a6b8c5576221ebca966affc94`).

Ultralytics/YOLO's AGPL obligations apply to its use and distribution. This bundle
does not grant an Enterprise license or permission to remove required source
availability, copyright or license notices from derivative deployments. Review
the upstream license if incorporating this application into a closed-source service.

The model bundle contains the exact public model bytes listed in
`configs/product/downloads.json`, these notices and license texts under
`docs/licenses/`. It contains no training optimizer, private dataset, user video,
evaluation label, API key or SSH credential. Historical Stage3A learned weights
are not required by the product and are not distributed.
