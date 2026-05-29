# GPU Inference Service

面向 Ubuntu + Docker + NVIDIA GPU 的 ONNX 推理服务。服务通过 FastAPI 暴露接口，按 GPU 拆成多个 worker，适合给人像识别、人像检索、ReID 等业务项目提供共享推理能力。

## 目录结构

```text
gpu-services/
├── Dockerfile
├── docker-compose.yml
├── main.py
└── requirements.txt
```

共享模型卷内的项目目录需要按下面格式放置：

```text
/workspace/projects/
└── your_project/
    └── models/
        └── your_model.onnx
```

## Ubuntu 服务器要求

- 已安装 NVIDIA 驱动，宿主机 `nvidia-smi` 正常。
- 已安装 Docker Engine 与 Docker Compose v2。
- 已安装 NVIDIA Container Toolkit。
- Docker 能运行 GPU 容器，例如：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 首次部署

创建共享模型卷：

```bash
docker volume create gpu-share-volume
```

把模型放入共享卷。下面命令只是示例，按你的实际项目名和模型文件调整：

```bash
docker run --rm -v gpu-share-volume:/projects -v "$PWD/models":/src ubuntu:22.04 \
  bash -lc "mkdir -p /projects/person_service/models && cp /src/*.onnx /projects/person_service/models/"
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
GET /models
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
POST /warmup
POST /reload
POST /unload
```

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

- 模型按 `project_name/model_name` 懒加载，第一次请求时从共享卷加载到当前 worker 的进程内存和 GPU 显存。
- 缓存是 worker 本地缓存，不在两个 worker 之间共享。同一个模型如果同时打到 `gpu-worker-0` 和 `gpu-worker-1`，会分别在两张 GPU 上各加载一份。
- 模型加载后默认不会自动卸载，直到容器重启或进程退出。这可以降低后续请求延迟，但会持续占用显存。
- 首次并发请求同一个模型时有加载锁，只有一个请求执行加载，其它请求等待加载完成后复用缓存。
- `MAX_LOADED_MODELS=0` 表示不限制缓存模型数量。设置为正整数后会启用 LRU 淘汰，超过上限时卸载最久未使用的模型。
- 可以通过 `WARMUP_MODELS` 在容器启动时预热模型，格式为逗号分隔的 `project/model.onnx`，例如 `person_service/reid.onnx,person_service/face.onnx`。
- `/unload` 可以手动卸载单个模型，`/reload` 可以在替换 ONNX 文件后强制重新加载。
- 如果替换了共享卷里的 ONNX 文件，已加载 worker 不会自动热更新。需要重启对应 worker 才能加载新模型：

```bash
docker compose restart gpu-worker-0
docker compose restart gpu-worker-1
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
- 当前接口使用 JSON 传输 tensor，简单通用但不是最高性能方案。如果单次输入很大或 QPS 很高，后续可考虑改成二进制协议、共享对象存储路径、gRPC，或让业务端只传图片路径并在服务端预处理。
- `/health` 表示服务进程正常，`/ready` 才表示 CUDA provider 可用。生产探活建议使用 `/ready`。

### 可观测性

- 每个 HTTP 请求都会返回 `X-Request-ID`。调用方也可以传入 `X-Request-ID`，服务会沿用该值。
- `/predict` 响应包含 `request_id`、是否冷加载、排队耗时、模型加载耗时、推理耗时、总耗时。
- 服务日志使用 JSON 字符串记录关键事件，包括 `http_request`、`predict_completed`、模型加载和模型卸载。
- `/metrics` 暴露 Prometheus 文本格式指标，包括请求量、推理失败数、模型加载数、缓存命中/未命中、已加载模型数、排队耗时总和、推理耗时总和。

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

- `PROJECTS_ROOT`: 模型共享目录，默认 `/workspace/projects`。
- `LOG_LEVEL`: 日志级别，默认 `INFO`。
- `MAX_TENSOR_ITEMS`: 单次请求最大 tensor 元素数，默认 `12582912`。
- `MAX_LOADED_MODELS`: 单 worker 最大缓存模型数，默认 `0` 表示不限制；正整数启用 LRU 淘汰。
- `GPU_QUEUE_LIMIT`: 单 worker 同时进入 GPU 推理段的请求数，默认 `1`。
- `WARMUP_MODELS`: 容器启动时自动预热的模型列表，格式为逗号分隔的 `project/model.onnx`。
- `API_TOKEN`: 可选接口令牌，留空时不启用鉴权；设置后 `/predict`、`/models`、`/model-info`、`/warmup`、`/reload`、`/unload` 需要携带令牌。
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
- 每个模型有独立推理锁，同一模型在单个 worker 内串行推理。
- 额外提供全局 GPU 推理信号量，避免不同模型同时挤占同一张 GPU。
- 路径使用 `Path.resolve()` 限制在共享模型目录内，避免路径穿越。
- `/ready` 会检查 `CUDAExecutionProvider` 是否可用。

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
