# GPU Inference Service

面向 Ubuntu + Docker + NVIDIA GPU 的 ONNX 推理服务。服务通过 FastAPI 暴露接口，按 GPU 拆成多个 worker，适合给人像识别、人像检索、ReID 等业务项目提供共享推理能力。

Ubuntu 服务器完整部署步骤见 [DEPLOY_UBUNTU.md](DEPLOY_UBUNTU.md)。

运行镜像基于 `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`，容器内使用 Python 3.12。当前依赖固定为 `onnxruntime-gpu==1.18.0`，需要 CUDA 11.x 运行库。

## 目录结构

```text
gpu-services/
├── app/
│   ├── constants.py
│   ├── core.py
│   ├── geometry.py
│   ├── image_io.py
│   ├── image_preprocess.py
│   ├── inference*.py
│   ├── metrics.py
│   ├── model_config*.py
│   ├── model_package.py
│   ├── model_refs.py
│   ├── observability.py
│   ├── postprocess.py
│   ├── runtime*.py
│   ├── security.py
│   ├── server.py
│   ├── settings.py
│   ├── schemas.py
│   ├── video_io.py
│   ├── vision.py
│   ├── routes.py
│   ├── routes_health.py
│   ├── routes_model*.py
│   ├── routes_vision.py
│   ├── routes_person*.py
│   └── routes_debug.py
├── Dockerfile
├── docker-compose.yml
├── main.py
├── models.yml
└── requirements.txt
```

`main.py` 只保留 `uvicorn main:app` 的兼容入口。`app/server.py` 只负责应用装配、middleware 和 startup。`app/routes.py` 是总路由聚合器，模型管理、人像检测/ReID/轨迹、通用视觉、健康检查和调试接口继续拆分到独立路由模块。共享能力按运行时、模型配置、模型包、图像/视频 IO、预处理、后处理、指标、安全和观测日志拆分，`app/core.py`、`app/runtime.py`、`app/inference.py`、`app/vision.py`、`app/model_config.py` 保留为兼容导出层。

共享模型目录默认与本项目目录同级。例如：

```text
~/project/
├── gpu-services/
├── other-project/
└── shared-models/
    └── your_project/
        └── your_model.onnx
```

## Ubuntu 服务器要求

- 已安装 NVIDIA 驱动，宿主机 `nvidia-smi` 正常。
- 已安装 Docker Engine 与 Docker Compose v2。
- 已安装 NVIDIA Container Toolkit。
- Docker 能运行 GPU 容器，例如：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 首次部署

创建共享模型目录：

```bash
mkdir -p ../shared-models
```

把模型放入共享目录。下面命令只是示例，按你的实际项目名和模型文件调整：

```bash
mkdir -p ../shared-models/person_service
cp "$PWD/models"/*.onnx ../shared-models/person_service/
```

构建并启动：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/ready
```

## 接口

健康检查：

```bash
GET /health
GET /ready
GET /ready/deep
GET /models
GET /model-configs
GET /metrics
GET /model-info?project_name=person_service&model_name=your_model.onnx
```

推理：

```bash
POST /predict
Content-Type: application/json

{
  "project_name": "person_service",
  "model_name": "your_model.onnx",
  "tensor_data": [[[[0.1, 0.2, 0.3]]]]
}
```

示例：

```bash
curl -X POST http://127.0.0.1:9001/predict \
  -H "Content-Type: application/json" \
  -d '{"project_name":"person_service","model_name":"your_model.onnx","tensor_data":[[[[0.1,0.2,0.3]]]]}'
```

响应：

```json
{
  "status": "success",
  "model": "person_service/your_model.onnx",
  "outputs": []
}
```

`outputs` 是 ONNX 模型的全部输出，按输出顺序返回二维或多维 list。

运维接口：

```bash
POST /infer/persons
POST /infer/person-embeddings
POST /infer/person-tracks
POST /infer/video/person-tracks
POST /infer/stream/person-tracks
POST /vision/infer
POST /vision/batch-infer
POST /debug/model-output
POST /warmup
POST /reload
POST /unload
POST /reload-config
GET /model-package
```

通用图像识别接口：

```bash
curl -X POST http://127.0.0.1:9001/vision/infer \
  -F "model_id=person_detector_default" \
  -F "files=@frame-001.jpg" \
  -F "confidence=0.25" \
  -F "iou=0.45"
```

`/vision/infer` 和 `/vision/batch-infer` 会按照 `models.yml` 中的 `task` 自动分派到检测、分类或 ReID 后处理。`model_id` 可以是 `aliases` 中的稳定别名，也可以直接使用 `project_name/model_name.onnx`。如果不使用别名，也可以传 `project_name` 和 `model_name`。单次请求默认最多 16 张图，可通过 `MAX_VISION_IMAGES` 调整。

多人检测接口：

```bash
curl -X POST http://127.0.0.1:9001/infer/persons \
  -F "project_name=cross_camera_tracking" \
  -F "model_name=yolov8n.onnx" \
  -F "confidence=0.25" \
  -F "iou=0.45" \
  -F "files=@frame-001.jpg" \
  -F "files=@frame-002.jpg"
```

`/infer/persons` 会在服务内完成图片解码、letterbox 预处理、YOLO 推理、person 类过滤和 NMS，只返回每帧的人体框，不再要求调用方解析 YOLO 原始 tensor。单次请求默认最多 16 张图，每张图默认最大 10MB，可通过 `MAX_PERSON_FRAMES` 和 `MAX_IMAGE_BYTES` 调整。

响应示例：

```json
{
  "status": "success",
  "model": "cross_camera_tracking/yolov8n.onnx",
  "frame_count": 2,
  "person_count": 3,
  "frames": [
    {
      "frame_index": 0,
      "filename": "frame-001.jpg",
      "width": 1920,
      "height": 1080,
      "person_count": 2,
      "persons": [
        {
          "box": [100.5, 80.2, 230.1, 420.9],
          "score": 0.91,
          "class_id": 0,
          "class_name": "person"
        }
      ]
    }
  ]
}
```

ReID 向量接口：

```bash
curl -X POST http://127.0.0.1:9001/infer/person-embeddings \
  -F "project_name=cross_camera_tracking" \
  -F "model_name=osnet_ibn_x1_0.onnx" \
  -F "include_vectors=true" \
  -F "files=@person-001.jpg" \
  -F "files=@person-002.jpg"
```

组合检测 + ReID 接口：

```bash
curl -X POST http://127.0.0.1:9001/infer/person-tracks \
  -F "detector_project_name=cross_camera_tracking" \
  -F "detector_model_name=yolov8n.onnx" \
  -F "reid_project_name=cross_camera_tracking" \
  -F "reid_model_name=osnet_ibn_x1_0.onnx" \
  -F "include_embeddings=false" \
  -F "files=@frame-001.jpg" \
  -F "files=@frame-002.jpg"
```

`/infer/person-tracks` 会先检测每帧人体，再裁剪人体并生成 ReID embedding。它不会伪造跨帧 `track_id`；调用方可以用返回的 `embedding_index`、`embedding_dim` 和可选 `embedding` 做自己的轨迹关联。

离线视频解析接口：

```bash
curl -X POST http://127.0.0.1:9001/infer/video/person-tracks \
  -F "file=@clip.mp4" \
  -F "frame_interval=15" \
  -F "max_frames=64" \
  -F "include_embeddings=false"
```

`/infer/video/person-tracks` 会上传视频文件、按帧间隔抽帧，再复用检测 + ReID 流水线。响应中的每帧会包含 `source_frame_index` 和可推导的 `source_seconds`。

视频流解析接口：

```bash
curl -X POST http://127.0.0.1:9001/infer/stream/person-tracks \
  -F "stream_url=rtsp://user:password@camera-host/stream1" \
  -F "frame_interval=15" \
  -F "max_frames=32" \
  -F "read_timeout_seconds=10"
```

`/infer/stream/person-tracks` 默认关闭，需要设置 `ALLOW_STREAM_URLS=true` 后才允许服务端主动拉取 RTSP/RTMP/HTTP/HTTPS 视频流。生产环境建议仅在可信内网启用，并通过网关限制可访问的摄像头地址。

模型输出调试接口：

```bash
curl -X POST http://127.0.0.1:9001/debug/model-output \
  -F "project_name=cross_camera_tracking" \
  -F "model_name=yolov8n.onnx" \
  -F "model_type=yolo" \
  -F "sample_values=12" \
  -F "file=@frame-001.jpg"
```

`/debug/model-output` 只返回输入 shape、输出 shape、min/max 和少量 sample 值，用于排查模型导出格式，不返回完整大 tensor。

预热示例：

```bash
curl -X POST http://127.0.0.1:9001/warmup \
  -H "Content-Type: application/json" \
  -d '{"models":[{"project_name":"person_service","model_name":"your_model.onnx"}]}'
```

模型元信息示例：

```bash
curl "http://127.0.0.1:9001/model-info?project_name=person_service&model_name=your_model.onnx"
```

`/model-info` 会返回输入名、输入 shape、输入 dtype、输出名、输出 shape、provider、模型 hash、文件大小、加载时间和推理次数。

## 运行机制与容量规划

### 并发模型

- `docker-compose.yml` 默认启动 2 个 worker：`gpu-worker-0` 绑定 GPU 0，`gpu-worker-1` 绑定 GPU 1。
- 每个 worker 内只启动 1 个 Uvicorn 进程，避免同一 GPU 上被多个进程重复加载模型。
- 每个 worker 使用 `--limit-concurrency 100` 限制进入服务层的并发请求数量。超过该限制时，Uvicorn 会拒绝或延迟处理新请求，调用方应设置合理超时。
- `GPU_QUEUE_LIMIT` 默认是 `1`，表示单 worker 内同一时间只允许 1 个请求进入 GPU 推理段。这样延迟更可控，也更适合显存较紧张的 2080 Ti。
- 同一个模型在同一个 worker 内使用独立推理锁串行执行，避免同一 ONNX Runtime session 被并发调用导致显存峰值或上下文竞争。
- 不同模型在同一个 worker 内各自有锁，代码层允许并发进入不同模型的推理逻辑，但实际 GPU 仍是共享资源；如果不同模型同时推理导致显存或延迟波动，应在业务侧固定路由或降低并发。

### 模型缓存

- 模型按 `project_name/model_name` 懒加载，第一次请求时从共享模型目录加载到当前 worker 的进程内存和 GPU 显存。
- 缓存是 worker 本地缓存，不在两个 worker 之间共享。同一个模型如果同时打到 `gpu-worker-0` 和 `gpu-worker-1`，会分别在两张 GPU 上各加载一份。
- 模型加载后默认不会自动卸载，直到容器重启或进程退出。这可以降低后续请求延迟，但会持续占用显存。
- 首次并发请求同一个模型时有加载锁，只有一个请求执行加载，其它请求等待加载完成后复用缓存。
- `MAX_LOADED_MODELS=0` 表示不限制缓存模型数量。设置为正整数后会启用 LRU 淘汰，超过上限时卸载最久未使用的模型。
- 可以通过 `WARMUP_MODELS` 在容器启动时预热模型，格式为逗号分隔的 `project/model.onnx`，例如 `person_service/reid.onnx,person_service/face.onnx`。
- `/unload` 可以手动卸载单个模型，`/reload` 可以在替换 ONNX 文件后强制重新加载。
- 如果替换了共享模型目录里的 ONNX 文件，已加载 worker 不会自动热更新。需要重启对应 worker 才能加载新模型：

```bash
docker compose restart gpu-worker-0
docker compose restart gpu-worker-1
```

### 模型配置

`models.yml` 用于声明业务模型类型、输入尺寸、后处理参数、模型包侧车文件和别名。没有配置的模型仍可通过 `/predict` 使用，但业务接口建议显式配置。新版配置兼容旧字段，例如 `type`、`input_size`、`confidence`、`iou` 仍然可以使用。

```yaml
aliases:
  person_detector_default:
    target: cross_camera_tracking/yolov8n.onnx

models:
  cross_camera_tracking/yolov8n.onnx:
    task: detection
    type: yolo
    runtime: onnxruntime
    version: 1.0.0
    precision: fp32
    input:
      size: [640, 640]
      layout: nchw
      dtype: float32
      color: rgb
      resize: letterbox
      normalize: none
    output:
      format: yolo
      classes: coco
      class_filter: [person]
      confidence: 0.25
      iou: 0.45
      max_detections: 100
    artifact:
      model_card: yolov8n.model-card.yml
      labels: yolov8n.labels.txt
      sha256: ""
  cross_camera_tracking/osnet_ibn_x1_0.onnx:
    task: reid
    type: reid
    input:
      size: [256, 128]
      normalize: imagenet
    output:
      format: embedding
      embedding_normalize: l2
```

可以通过 `/model-configs` 查看当前加载的配置和别名，通过 `/reload-config` 在不重启容器的情况下重新读取配置，通过 `/model-package` 查看模型卡、labels、sha256 匹配状态等模型包信息。

### 测试与上线校验

本项目提供三类工程校验：

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python tools/deploy_check.py --import-app
python tools/validate_model_package.py --config models.yml --models-root ../shared-models
```

- `pytest` 覆盖 API 契约、路径安全、模型配置兼容解析、检测/分类/ReID 后处理和模型包校验脚本。
- `tools/deploy_check.py` 用于部署前静态检查，会验证关键文件、Python 语法、`models.yml`、Docker Compose GPU 配置和核心路由。
- `tools/validate_model_package.py` 用于上线前校验算法侧交付的模型包。生产上线建议加 `--strict-hash --strict-sidecars`，要求 sha256、模型卡和 labels 齐全。
- `tools/regression_check.py` 用于固定样例回归检查，可以对运行中的服务发起 HTTP 请求，并按期望输出子集和浮点容忍阈值比对。

服务启动后可以执行 HTTP smoke test：

```bash
python tools/service_smoke_test.py \
  --base-url http://127.0.0.1:9001 \
  --token "$API_TOKEN" \
  --require-ready \
  --model-id person_detector_default
```

如果只是本地开发且没有 CUDA，可以不加 `--require-ready`，此时 `/ready` 返回 503 不会作为硬失败。真实上线前应在 GPU 服务器上执行 `--require-ready`，并按需增加 `--deep-ready --load-models --dummy-inference`。

固定回归集 manifest 示例：

```yaml
tolerance: 0.001
cases:
  - name: health_contract
    method: GET
    path: /health
    expected:
      status: healthy

  - name: detector_sample
    method: POST
    path: /vision/infer
    form:
      model_id: person_detector_default
      confidence: "0.25"
      iou: "0.45"
    files:
      files: samples/frame_001.jpg
    expected_path: expected/frame_001.expected.json
```

执行：

```bash
python tools/regression_check.py \
  --manifest regression.yml \
  --base-url http://127.0.0.1:9001 \
  --token "$API_TOKEN"
```

多 worker 运维控制可以使用：

```bash
python tools/worker_control.py --action health
python tools/worker_control.py --action reload-config --token "$API_TOKEN"
python tools/worker_control.py --action warmup --token "$API_TOKEN" --model cross_camera_tracking/yolov8n.onnx
```

### 上线切换和回滚

`models.yml` 的 `aliases` 用于稳定暴露模型入口，例如 `person_detector_default`。新模型上线建议流程：

1. 把新模型包放入共享模型目录。
2. 在 `models.yml` 的 `models` 中增加新模型配置，`rollout.status` 先写 `candidate`。
3. 执行模型包校验和 smoke test。
4. 使用别名切换接口把默认别名指向新模型。

查看别名：

```bash
curl -H "Authorization: Bearer $API_TOKEN" \
  http://127.0.0.1:9001/rollout/aliases
```

dry-run 切换：

```bash
curl -X POST http://127.0.0.1:9001/rollout/aliases/switch \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "alias_name": "person_detector_default",
    "target_model_id": "cross_camera_tracking/person_detector_yolov8n_v1.1.0_fp32.onnx",
    "expected_current_target": "cross_camera_tracking/yolov8n.onnx",
    "dry_run": true
  }'
```

确认后把 `dry_run` 改为 `false`。服务会写回宿主机挂载的 `models.yml`，并重新加载当前 worker 的配置；其它 worker 可通过 `tools/worker_control.py --action reload-config` 同步新配置。回滚到上一个目标：

```bash
curl -X POST http://127.0.0.1:9001/rollout/aliases/rollback \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"alias_name":"person_detector_default","dry_run":false}'
```

按权重灰度时，可以把同一个别名配置成多目标分流。`traffic_key` 相同的请求会稳定命中同一个目标；如果不传 `traffic_key`，服务会使用请求 ID：

```bash
curl -X POST http://127.0.0.1:9001/rollout/aliases/weighted \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "alias_name": "person_detector_default",
    "expected_current_target": "cross_camera_tracking/yolov8n.onnx",
    "dry_run": true,
    "targets": [
      {
        "target_model_id": "cross_camera_tracking/yolov8n.onnx",
        "weight": 90,
        "status": "active"
      },
      {
        "target_model_id": "cross_camera_tracking/person_detector_yolov8n_v1.1.0_fp32.onnx",
        "weight": 10,
        "status": "candidate"
      }
    ]
  }'
```

预览某个流量 key 会命中的目标：

```bash
curl -H "Authorization: Bearer $API_TOKEN" \
  "http://127.0.0.1:9001/rollout/aliases/preview?alias_name=person_detector_default&traffic_key=customer-001"
```

业务调用 `/vision/infer` 时也可以传 `traffic_key`：

```bash
curl -X POST http://127.0.0.1:9001/vision/infer \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "model_id=person_detector_default" \
  -F "traffic_key=customer-001" \
  -F "files=@frame-001.jpg"
```

### 调用方建议

- 低延迟场景建议业务侧固定访问某一个 worker，避免同一模型在多张卡上重复冷启动。
- 高吞吐场景可以在业务侧或网关侧按 GPU worker 做负载均衡，但要接受每张卡各自缓存模型带来的显存占用。
- 调用方应设置连接超时和读取超时；读取超时需要覆盖“排队时间 + 推理时间 + 首次模型加载时间”。
- 推理接口不是幂等写操作，但计算结果可重复；网络超时后的重试可能造成同一个请求被重复推理，业务侧需要自行去重或接受重复计算成本。
- 大 tensor 会增加 JSON 解析、网络传输和内存复制成本。`MAX_TENSOR_ITEMS` 只限制元素数量，不限制 HTTP body 字节数；生产环境建议在反向代理层额外限制请求体大小。

### 显存与性能影响

- 显存主要由已加载模型、ONNX Runtime CUDA arena、输入输出 tensor、并发中的临时 buffer 决定。
- `gpu_mem_limit` 当前为 `0`，表示由 ONNX Runtime 自动管理显存。显存紧张时，优先减少单个 worker 上加载的模型数量或降低 batch size。
- `GPU_QUEUE_LIMIT` 控制单 worker 同时进入 GPU 段的请求数量，`MODEL_CONCURRENCY_LIMIT` 控制单模型默认并发，模型配置里的 `max_concurrency` 或 `runtime.max_concurrency` 可以单独覆盖。
- `MODEL_QUEUE_TIMEOUT_SECONDS` 大于 0 时，请求在模型队列或 GPU 队列等待超过该秒数会返回 503，避免无限堆积。
- FP16 ONNX 输入会按模型输入 dtype 自动 cast；如果模型输入是 `tensor(float16)`，图像预处理结果会在进入 session 前转为 `float16`。
- 配置 `runtime: tensorrt` 且 `ENABLE_TENSORRT=true` 时，服务会优先请求 ONNX Runtime 的 `TensorrtExecutionProvider`。运行环境必须实际包含该 provider，否则模型加载会失败并给出清晰错误。
- 当前接口使用 JSON 传输 tensor，简单通用但不是最高性能方案。如果单次输入很大或 QPS 很高，后续可考虑改成二进制协议、共享对象存储路径、gRPC，或让业务端只传图片路径并在服务端预处理。
- `/health` 表示服务进程正常，`/ready` 才表示 CUDA provider 可用。生产探活建议使用 `/ready`。

### 可观测性

- 每个 HTTP 请求都会返回 `X-Request-ID`。调用方也可以传入 `X-Request-ID`，服务会沿用该值。
- `/predict` 和业务接口响应包含 `request_id`、是否冷加载、排队耗时、模型加载耗时、推理耗时、总耗时。
- 业务接口会额外记录 `decode_seconds`、`preprocess_seconds`、`postprocess_seconds`、`frame_count`、`person_count` 和 `inference_mode`。
- 服务日志使用 JSON 字符串记录关键事件，包括 `http_request`、`predict_completed`、`persons_infer_completed`、`embeddings_infer_completed`、`person_tracks_infer_completed`、模型加载和模型卸载。
- `/metrics` 暴露 Prometheus 文本格式指标，包括请求量、推理失败数、模型加载数、缓存命中/未命中、已加载模型数、排队耗时总和、推理耗时总和、图片解码耗时、预处理耗时、后处理耗时、检测人数和处理帧数。
- `/metrics` 还暴露模型维度指标，例如 `gpu_worker_model_config_info`、`gpu_worker_model_loaded_info`、`gpu_worker_model_inference_count_total`，标签包含 `model`、`task`、`version` 和 `status`。
- 别名切换、weighted rollout 和 rollback 会追加写入 `ROLLOUT_AUDIT_PATH` 指向的 JSONL 文件。Docker Compose 默认把审计文件放在宿主机 `./runtime-state/` 中，便于容器重建后继续保留。

## 业务容器接入

业务容器如果和本服务在同一台 Docker 主机上，建议加入同一个网络：

```yaml
networks:
  gpu-bridge:
    external: true
```

然后通过容器名调用：

```text
http://gpu-worker-0:8000/predict
http://gpu-worker-1:8000/predict
```

宿主机本地调试端口：

- GPU 0 worker: `http://127.0.0.1:9001`
- GPU 1 worker: `http://127.0.0.1:9002`

端口默认只绑定 `127.0.0.1`，外部机器不能直接访问。需要跨机器访问时，建议放在受控内网网关或反向代理后面，再加鉴权和限流。

## 配置项

通过 `docker-compose.yml` 的环境变量调整：

- `MODELS_HOST_DIR`: 宿主机模型共享目录，默认 `../shared-models`，即本项目同级目录。
- `MODELS_ROOT`: 容器内模型目录，固定为 `/models`。
- `MODEL_CONFIG_HOST_FILE`: 宿主机模型配置文件，默认 `./models.yml`，Compose 会可写挂载到容器内。
- `MODEL_CONFIG_PATH`: 容器内模型配置文件路径，默认 `/workspace/models.yml`；本地直接运行默认读取当前目录 `models.yml`。
- `LOG_LEVEL`: 日志级别，默认 `INFO`。
- `MAX_TENSOR_ITEMS`: 单次请求最大 tensor 元素数，默认 `12582912`。
- `MAX_LOADED_MODELS`: 单 worker 最大缓存模型数，默认 `0` 表示不限制；正整数启用 LRU 淘汰。
- `GPU_QUEUE_LIMIT`: 单 worker 同时进入 GPU 推理段的请求数，默认 `1`。
- `MODEL_CONCURRENCY_LIMIT`: 单模型默认并发限制，默认 `1`。
- `MODEL_QUEUE_TIMEOUT_SECONDS`: 模型队列和 GPU 队列等待超时秒数，默认 `0` 表示不超时。
- `ENABLE_TENSORRT`: 是否允许 `runtime: tensorrt` 模型使用 TensorRT Execution Provider，默认 `false`。
- `TENSORRT_ENGINE_CACHE_ENABLE`: 是否启用 TensorRT engine cache，默认 `true`。
- `TENSORRT_ENGINE_CACHE_PATH`: TensorRT engine cache 路径，默认 `/tmp/tensorrt-engine-cache`。
- `RUNTIME_STATE_HOST_DIR`: 宿主机运行期状态目录，默认 `./runtime-state`。
- `ROLLOUT_AUDIT_PATH`: 灰度/别名变更审计 JSONL 文件路径，默认 `/workspace/runtime-state/rollout-audit.jsonl`。
- `MAX_IMAGE_BYTES`: `/infer/persons` 单张上传图片大小上限，默认 `10485760`。
- `MAX_PERSON_FRAMES`: `/infer/persons` 单次请求图片数量上限，默认 `16`。
- `MAX_EMBEDDING_IMAGES`: `/infer/person-embeddings` 单次请求图片数量上限，默认 `64`。
- `MAX_PIPELINE_FRAMES`: `/infer/person-tracks` 单次请求帧数量上限，默认 `16`。
- `MAX_VIDEO_BYTES`: `/infer/video/person-tracks` 单个视频文件大小上限，默认 `104857600`。
- `VIDEO_FRAME_INTERVAL`: 离线视频默认抽帧间隔，默认 `15`。
- `MAX_VIDEO_FRAMES`: 离线视频单次最多抽取帧数，默认 `64`。
- `STREAM_FRAME_INTERVAL`: 视频流默认抽帧间隔，默认 `15`。
- `MAX_STREAM_FRAMES`: 视频流单次最多抽取帧数，默认 `32`。
- `STREAM_READ_TIMEOUT_SECONDS`: 视频流单次读取软超时，默认 `10`。
- `ALLOW_STREAM_URLS`: 是否允许服务端主动拉取视频流 URL，默认 `false`。
- `WARMUP_MODELS`: 容器启动时自动预热的模型列表，格式为逗号分隔的 `project/model.onnx`。
- `API_TOKEN`: 可选接口令牌，留空时不启用鉴权；设置后业务接口、调试接口、模型管理接口和深度 ready 需要携带令牌。
- `NVIDIA_VISIBLE_DEVICES`: 当前 worker 可见 GPU。
- `NVIDIA_DRIVER_CAPABILITIES`: 默认 `compute,utility`。

Uvicorn 启动参数在 `Dockerfile` 的 `CMD` 中配置：

- `--workers 1`: 每个容器单进程运行，保证进程内模型缓存和锁有效。
- `--limit-concurrency 100`: 限制单 worker 的服务层并发数量，可按 GPU 性能和业务超时调整。

启用鉴权后的请求示例：

```bash
curl -X POST http://127.0.0.1:9001/predict \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"project_name":"person_service","model_name":"your_model.onnx","tensor_data":[[[[0.1,0.2,0.3]]]]}'
```

## 设计说明

- 每个 worker 只运行一个 Uvicorn 进程，避免多进程重复加载同一模型占用显存。
- 模型按 `project_name/model_name` 懒加载并缓存。
- 首次加载同一模型时使用加载锁，避免并发请求重复加载模型。
- 支持启动预热、手动预热、手动卸载、手动重载和 LRU 缓存上限。
- 支持模型配置文件，把 YOLO/ReID 的输入尺寸、类别 ID 和归一化策略显式化。
- 每个模型有独立 semaphore，默认串行，也可以通过模型配置或环境变量提高并发。
- 额外提供全局 GPU 推理信号量，避免不同模型同时挤占同一张 GPU。
- 支持 FP16 输入自动 cast，支持按模型配置启用 TensorRT Execution Provider。
- 提供 `tools/worker_control.py` 对多个 worker 统一执行健康检查、配置重载、预热、重载和卸载。
- 路径使用 `Path.resolve()` 限制在共享模型目录内，避免路径穿越。
- `/ready` 会检查 `CUDAExecutionProvider` 是否可用，`/ready/deep` 可进一步检查配置模型、加载模型和 dummy inference。

## 压测记录模板

上线前建议为每个模型记录一次压测结果：

| 项目 | 数值 |
| --- | --- |
| 模型 | `person_service/your_model.onnx` |
| GPU | 例如 `RTX 2080 Ti 11GB` |
| 输入 shape | 例如 `[1, 3, 256, 128]` |
| batch size | 例如 `1` |
| 冷启动耗时 | 例如 `2.3s` |
| 热缓存平均延迟 | 例如 `18ms` |
| P95 / P99 | 例如 `35ms / 60ms` |
| 稳定 QPS | 例如 `40` |
| 单模型显存占用 | 例如 `1.2GB` |
| 推荐 `GPU_QUEUE_LIMIT` | 例如 `1` |

## 常见问题

如果 `/ready` 返回没有 `CUDAExecutionProvider`：

1. 确认宿主机 `nvidia-smi` 正常。
2. 确认 NVIDIA Container Toolkit 已安装。
3. 确认 `docker run --rm --gpus all ... nvidia-smi` 正常。
4. 确认 `docker compose` 版本支持 GPU device reservation。

如果显存不足：

1. 减少同时加载的模型数量。
2. 降低输入 batch size。
3. 将模型拆到不同 worker 或不同 GPU。
4. 考虑导出 FP16 或 TensorRT 版本。
