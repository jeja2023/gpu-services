# 图像识别推理服务升级扩展方案

本文档面向当前 `gpu-services` 项目，目标是把现有 GPU ONNX 推理服务升级为可持续扩展的图像识别推理平台。训练、标注、模型研发不放进本服务内，而由配套方案 `MODEL_RND_TRAINING_PLAN.md` 承接。两套方案通过统一的模型交付契约、版本规则、评估口径和上线流程匹配。

## 1. 当前基础判断

当前项目已经具备继续升级的基础：

- 服务框架：FastAPI + Uvicorn。
- 推理运行时：ONNX Runtime GPU。
- 部署方式：Docker + NVIDIA GPU + Docker Compose 多 worker。
- 模型来源：共享模型目录 `shared-models`。
- 模型加载：按 `project_name/model_name` 懒加载，支持缓存、卸载、重载和预热。
- 图像能力：已包含图片解码、YOLO letterbox、NMS、ReID resize/normalize、视频抽帧、流地址抽帧。
- 运维能力：健康检查、ready/deep ready、Prometheus 文本指标、JSON 日志、API token 鉴权。

主要限制：

- 业务逻辑集中在 `main.py`，继续扩展会很快变得难维护。
- 模型类型目前以 `yolo`、`reid` 为主，还没有通用任务插件机制。
- `models.yml` 能描述基础参数，但还不能完整表达版本、标签、精度、输入输出契约、验收样例和灰度策略。
- 缺少自动化测试、模型上线前验证、回归集评估和模型包校验。
- 通用 `/predict` 会返回完整 tensor，不适合大输出模型直接暴露给业务方。

结论：本项目适合升级为“视觉模型推理服务/推理平台”，不适合承载训练、标注、实验管理等算法研发活动。

## 2. 目标定位

本项目负责：

- 加载算法侧交付的标准模型包。
- 执行图片、视频帧、视频流的推理。
- 提供稳定的业务 API。
- 管理模型缓存、预热、重载、灰度和回滚。
- 记录性能、错误、模型版本和请求链路。
- 提供模型上线前的服务侧验证。

本项目不负责：

- 数据采集和标注管理。
- 模型训练、微调、蒸馏和调参。
- 实验追踪和训练资源调度。
- 大规模离线评估集管理。
- 模型导出脚本长期维护。

这些内容由 `MODEL_RND_TRAINING_PLAN.md` 中的算法研发体系承接。

## 3. 两套方案的匹配方式

训练研发侧交付“标准模型包”，推理服务侧消费“标准模型包”。匹配点如下：

| 匹配项 | 算法研发侧负责 | 推理服务侧负责 |
| --- | --- | --- |
| 模型格式 | 导出 ONNX，必要时提供 TensorRT 构建建议 | 加载 ONNX，后续可按环境构建/加载 TensorRT |
| 输入契约 | 声明输入 shape、dtype、颜色顺序、归一化 | 按契约执行预处理并校验请求 |
| 输出契约 | 声明输出 tensor 语义、类别、阈值建议 | 按任务插件执行后处理并返回业务结构 |
| 版本 | 生成语义化版本和模型卡 | 按版本加载、预热、灰度、回滚 |
| 验收样例 | 提供样例图片、期望输出、误差容忍 | 上线前自动执行 smoke test 和回归验证 |
| 指标 | 提供算法指标、测试集、阈值 | 记录线上延迟、错误率、吞吐和输出统计 |
| 上线审批 | 给出模型评估结论 | 执行服务兼容性验证并发布 |

短期保持当前路径兼容：

```text
shared-models/
  cross_camera_tracking/
    person_detector_yolov8n_v1.0.0_fp32.onnx
    person_detector_yolov8n_v1.0.0_fp32.model-card.yml
    person_detector_yolov8n_v1.0.0_fp32.labels.txt
```

当前服务仍通过：

```text
project_name = cross_camera_tracking
model_name   = person_detector_yolov8n_v1.0.0_fp32.onnx
```

中期再升级为模型注册表/别名方式：

```yaml
aliases:
  person_detector_default:
    project_name: cross_camera_tracking
    model_name: person_detector_yolov8n_v1.0.0_fp32.onnx
```

这样业务方可以调用稳定别名，服务内部做版本切换。

## 4. 推荐目录结构

第一阶段建议从单文件拆分为如下结构：

```text
gpu-services/
  app/
    __init__.py
    main.py
    config.py
    auth.py
    metrics.py
    logging_utils.py
    schemas.py
    api/
      health.py
      models.py
      predict.py
      vision.py
      debug.py
    runtime/
      registry.py
      onnx_runtime.py
      tensorrt_runtime.py
      model_package.py
      warmup.py
    vision/
      image_io.py
      video_io.py
      preprocess.py
      geometry.py
      nms.py
    tasks/
      base.py
      yolo_detection.py
      classification.py
      segmentation.py
      reid.py
      ocr.py
    validation/
      smoke_test.py
      contract_check.py
      regression.py
  tests/
    test_preprocess.py
    test_nms.py
    test_model_config.py
    test_api_contract.py
  main.py
  models.yml
  requirements.txt
```

`main.py` 可以先保留为兼容入口，只负责导入 `app.main:app`。这能降低部署脚本和 Dockerfile 的改动风险。

## 5. 模型配置升级

当前 `models.yml` 可以继续使用，但需要逐步扩展字段。建议目标结构如下：

```yaml
models:
  cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx:
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
      sha256: ""
      model_card: person_detector_yolov8n_v1.0.0_fp32.model-card.yml
      labels: person_detector_yolov8n_v1.0.0_fp32.labels.txt
    rollout:
      status: candidate
      warmup: true
```

字段含义：

- `task`：面向业务的任务类型，例如 `classification`、`detection`、`segmentation`、`reid`、`ocr`。
- `type`：具体后处理实现，例如 `yolo`、`resnet_classifier`、`deeplab`、`clip_embedding`。
- `runtime`：`onnxruntime` 为主，后续可扩展 `tensorrt`。
- `version`：模型版本，和算法侧模型卡一致。
- `precision`：`fp32`、`fp16`、`int8`。
- `input`：预处理契约。
- `output`：后处理契约。
- `artifact`：模型包元信息。
- `rollout`：上线状态和预热策略。

短期改造原则：

- 不破坏当前 `type: yolo`、`type: reid` 配置。
- 新字段可选，未配置时使用当前默认值。
- `/model-configs` 返回完整解析后的配置，便于业务侧确认。

## 6. 任务插件设计

新增模型类型不应该继续堆到接口函数里，而应通过任务插件完成。

统一接口建议：

```python
class VisionTask:
    task_name: str

    def build_input(self, images, config) -> tuple[np.ndarray, dict]:
        ...

    def postprocess(self, outputs, context, config) -> dict:
        ...
```

每个任务插件负责：

- 读取模型配置。
- 执行预处理。
- 调用 runtime。
- 执行后处理。
- 返回统一响应结构。
- 提供 smoke test 所需的最小样例校验。

优先级：

1. `yolo_detection`：兼容当前 `/infer/persons`。
2. `reid`：兼容当前 `/infer/person-embeddings`。
3. `classification`：图片分类、属性识别、质量判断。
4. `segmentation`：语义/实例分割，返回 mask、polygon 或 RLE。
5. `ocr`：文本检测 + 文本识别，可先支持单模型识别，再支持流水线。
6. `multi_stage_pipeline`：检测 + 分类、检测 + ReID、检测 + OCR 等组合。

## 7. API 升级路线

保留现有接口：

- `GET /health`
- `GET /ready`
- `GET /ready/deep`
- `GET /models`
- `GET /model-configs`
- `GET /model-info`
- `POST /predict`
- `POST /infer/persons`
- `POST /infer/person-embeddings`
- `POST /infer/person-tracks`
- `POST /infer/video/person-tracks`
- `POST /infer/stream/person-tracks`
- `POST /warmup`
- `POST /reload`
- `POST /unload`

新增通用视觉接口：

```text
POST /vision/infer
POST /vision/batch-infer
POST /vision/video-infer
POST /vision/stream-infer
```

`/vision/infer` 示例：

```bash
curl -X POST http://127.0.0.1:9001/vision/infer \
  -F "model_id=person_detector_default" \
  -F "file=@frame.jpg"
```

响应建议：

```json
{
  "status": "success",
  "request_id": "xxx",
  "model": {
    "id": "person_detector_default",
    "project_name": "cross_camera_tracking",
    "model_name": "person_detector_yolov8n_v1.0.0_fp32.onnx",
    "version": "1.0.0"
  },
  "task": "detection",
  "results": [
    {
      "image_index": 0,
      "width": 1920,
      "height": 1080,
      "detections": []
    }
  ],
  "timing": {
    "decode_seconds": 0.0,
    "preprocess_seconds": 0.0,
    "queue_seconds": 0.0,
    "inference_seconds": 0.0,
    "postprocess_seconds": 0.0,
    "total_seconds": 0.0
  }
}
```

迁移策略：

- 现有业务继续使用旧接口。
- 新业务优先接入 `/vision/infer`。
- 旧接口内部逐步复用任务插件，避免维护两套逻辑。

## 8. Runtime 策略

短期主线：

- 默认使用 ONNX Runtime GPU。
- 保留 CPU fallback 只用于本地开发或诊断，不建议生产默认启用。
- 每个 worker 绑定一张 GPU。
- 模型按需懒加载，重要模型启动时预热。

中期增强：

- 支持按模型配置选择 runtime。
- 支持 ONNX Runtime provider 参数配置。
- 支持 TensorRT engine 构建/加载，但不替代 ONNX 主格式。
- 支持模型级并发限制，避免所有模型共享同一个粗粒度 GPU 队列。

长期增强：

- 针对大模型或高吞吐模型增加动态 batch。
- 引入异步队列和批处理窗口。
- 按模型延迟、显存、QPS 自动调整 worker 路由。
- 对多 GPU 服务增加统一网关或服务发现。

## 9. 模型上线流程

标准流程：

1. 算法侧提交模型包到共享模型目录或模型制品仓库。
2. 推理服务读取模型卡和 `models.yml`。
3. 执行模型包校验：
   - ONNX 文件存在。
   - sha256 匹配。
   - 输入输出 shape 可解析。
   - labels 文件存在。
   - 样例输入可推理。
4. 执行服务侧 smoke test：
   - 加载模型。
   - 执行一张样例图片。
   - 检查输出字段。
   - 检查延迟阈值。
5. 执行回归测试：
   - 使用算法侧提供的小型固定评估集。
   - 比较核心指标和关键样本结果。
6. 预热候选模型。
7. 通过别名或配置灰度切换。
8. 观察线上指标。
9. 若错误率或延迟异常，回滚别名到旧版本。

模型状态建议：

- `candidate`：候选，允许测试，不默认对外。
- `staging`：预发布，可灰度。
- `active`：线上默认。
- `deprecated`：不再推荐使用。
- `blocked`：校验失败或禁止上线。

## 10. 灰度和回滚

灰度方案从简单到复杂分三步：

第一步：手动别名切换。

```yaml
aliases:
  person_detector_default:
    target: cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx
```

第二步：按比例灰度。

```yaml
aliases:
  person_detector_default:
    rollout:
      - target: cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx
        weight: 90
      - target: cross_camera_tracking/person_detector_yolov8n_v1.1.0_fp16.onnx
        weight: 10
```

第三步：按请求标签灰度。

- 指定客户。
- 指定摄像头。
- 指定业务项目。
- 指定 `X-Model-Version`。

回滚原则：

- 只切换别名，不删除模型。
- 保留至少一个稳定版本常驻预热。
- 新版本上线观察期内保留旧版本缓存。

## 11. 监控指标

已有指标应保留，并新增模型维度指标。

服务指标：

- 请求总数。
- 错误总数。
- p50/p95/p99 总延迟。
- 解码、预处理、排队、推理、后处理耗时。
- 当前加载模型数量。
- 模型加载次数和失败次数。
- GPU 队列等待时间。

模型指标：

- `model_id`。
- `model_version`。
- `runtime`。
- `precision`。
- `input_shape`。
- `output_shape`。
- 每模型 QPS。
- 每模型错误率。
- 每模型冷加载次数。
- 每模型平均输出数量，例如检测框数量、分类置信度分布。

业务指标：

- 每帧检测数量。
- 低置信度比例。
- 空结果比例。
- 视频抽帧数量。
- 流读取失败率。

## 12. 安全和稳定性

必须保留：

- API token 鉴权。
- 模型路径校验，禁止路径穿越。
- 上传图片大小限制。
- 视频大小和抽帧数量限制。
- 流地址默认关闭。

建议新增：

- `/predict` 默认不对公网开放，或通过配置关闭完整 tensor 输出。
- 对 `stream_url` 增加内网白名单，避免 SSRF。
- 对返回 embedding 的接口增加向量数量和维度限制。
- 模型包 sha256 校验。
- 模型配置 schema 校验。
- 单模型超时和熔断。
- 请求体和响应体大小限制。

## 13. 测试计划

单元测试：

- 路径校验。
- `models.yml` 解析。
- 图片 resize、letterbox、normalize。
- bbox 坐标恢复。
- NMS。
- embedding normalize。
- 视频抽帧参数校验。

集成测试：

- 使用小型 ONNX dummy 模型测试加载、推理、卸载、重载。
- 测试 `/health`、`/ready`、`/models`、`/model-info`。
- 测试 `/vision/infer` 的分类、检测、ReID 基础响应。
- 测试鉴权。

模型上线测试：

- smoke test。
- 样例图片输出格式校验。
- 与算法侧期望输出比对。
- 延迟和显存基线记录。

生产回归测试：

- 固定评估集。
- 每个任务保留 20 到 200 张代表样例。
- 输出差异超过阈值时阻止上线。

当前进展：

- 已新增 `tools/regression_check.py`，支持 YAML/JSON manifest、multipart 图片上传、期望输出子集比对和浮点容忍阈值。
- 当前测试集已覆盖 API contract、配置解析、路径安全、后处理算法、模型包校验、别名切换和回归比较逻辑。

## 14. 分阶段实施计划

### 第 0 阶段：整理当前服务

目标：不改变行为，降低维护风险。

工作：

- 增加测试依赖和最小测试目录。
- 为当前 `models.yml` 增加 schema 校验。
- 把 README 中的模型配置和接口契约整理为稳定文档。
- 梳理当前接口响应字段，形成 API contract。

验收：

- 当前接口行为不变。
- Docker 构建不变。
- 至少有基础单元测试。

当前进展：

- 已新增 `requirements-dev.txt`、`pytest.ini` 和 `tests/` 基础测试目录。
- 已覆盖 API contract、路径校验、模型配置兼容解析、NMS、YOLO person 后处理、分类后处理、embedding normalize 和模型包校验脚本。
- 已新增 `tools/deploy_check.py`，用于部署前检查关键文件、Python 语法、`models.yml`、Docker Compose GPU 配置和核心路由。

### 第 1 阶段：模块化拆分

目标：把单文件服务拆成可扩展结构。

工作：

- 拆出 `config.py`、`auth.py`、`metrics.py`。
- 拆出 `runtime/registry.py` 和 `runtime/onnx_runtime.py`。
- 拆出 `vision/image_io.py`、`vision/preprocess.py`、`vision/nms.py`。
- 拆出 `tasks/yolo_detection.py`、`tasks/reid.py`。
- 保留原有接口路径。

当前进展：

- 根目录 `main.py` 已缩减为兼容入口。
- 服务装配已迁移到 `app/server.py`。
- `app/routes.py` 已缩减为路由聚合器。
- 健康检查、模型管理、模型生命周期、原始 tensor 推理、通用视觉、人像检测、ReID embedding、图像轨迹、视频轨迹、流式轨迹和调试接口已拆分到独立 `routes_*.py` 模块。
- 运行时加载已拆分为状态、会话、缓存注册和执行模块。
- 模型配置已拆分为加载、状态、访问和引用解析模块。
- 视觉处理已拆分为图片 IO、视频 IO、预处理、几何/NMS 和后处理模块。
- 推理编排已按检测、ReID、分类和组合轨迹拆分。
- `app/core.py`、`app/runtime.py`、`app/inference.py`、`app/vision.py`、`app/model_config.py` 已缩减为兼容导出层，便于后续继续调整引用关系。
- 环境配置已拆分到 `app/settings.py`。
- 请求和内部类型已拆分到 `app/schemas.py`。
- 模型路径和 cache key 工具已拆分到 `app/model_refs.py`。

验收：

- 旧接口全部可用。
- 预处理和后处理测试通过。
- 模型加载、预热、卸载、重载行为不变。

### 第 2 阶段：标准模型包和通用视觉接口

目标：让服务能消费算法侧标准交付物。

工作：

- 扩展 `models.yml` 字段。
- 支持模型卡和 labels 文件读取。
- 增加模型包校验。
- 增加 `/vision/infer` 和 `/vision/batch-infer`。
- 增加分类插件。

验收：

- 能通过 `model_id` 或当前 `project/model` 加载模型。
- 分类、检测、ReID 至少三类任务可通过统一接口调用。
- 模型包缺字段时能给出清晰错误。

当前进展：

- `models.yml` 已扩展 `aliases`、`task`、`input`、`output`、`artifact` 和 `rollout` 字段。
- 已支持模型卡、labels、sha256 配置和 `/model-package` 查询。
- 已新增 `tools/validate_model_package.py`，可在上线前检查模型文件、模型卡、labels、sha256 和别名目标。
- 已新增通用 `/vision/infer` 和 `/vision/batch-infer`，支持检测、分类和 ReID 分发。

### 第 3 阶段：上线治理

目标：支持候选、预发布、上线和回滚。

工作：

- 增加模型状态。
- 增加模型别名。
- 增加 smoke test 命令或接口。
- 增加灰度切换配置。
- 增加模型维度指标。

验收：

- 新模型可先作为 candidate 预热和测试。
- active 别名可切换到新版本。
- 旧版本可快速回滚。
- 监控能区分模型版本。

当前进展：

- `models.yml` 已支持别名和 `rollout.status` 配置，能描述 active/candidate 等上线状态。
- 已提供 `/reload-config`、`/warmup`、`/reload`、`/unload`，支持配置重载、预热、重载和手动回滚。
- 已新增 `tools/service_smoke_test.py`，支持 `/health`、`/ready`、OpenAPI、`/metrics`、`/ready/deep` 和 `/model-package` 检查。
- 已新增 `/rollout/aliases`、`/rollout/aliases/preview`、`/rollout/aliases/switch`、`/rollout/aliases/weighted`、`/rollout/aliases/rollback`，支持别名查看、灰度预览、dry-run 切换、按权重分流、乐观校验和回滚到 previous target。
- `/vision/infer` 已支持 `traffic_key`，用于加权灰度时做稳定 hash 分流；不传时使用请求 ID。
- `/metrics` 已增加模型维度指标，标签包含 `model`、`task`、`version` 和 `status`。
- 已新增 `tools/regression_check.py`，支持固定回归集比对。
- 多 worker 统一控制面、灰度审计记录和更细的模型指标 label 仍作为后续增强项。

### 第 4 阶段：性能优化

目标：在稳定基础上提升吞吐和降低延迟。

工作：

- 增加模型级队列和并发限制。
- 支持 FP16 ONNX。
- 增加 TensorRT engine 支持。
- 增加动态 batch 或批处理窗口。
- 增加多 worker 路由策略。

验收：

- 关键模型达到约定 p95 延迟。
- GPU 显存峰值可控。
- TensorRT 版本和 ONNX 版本结果差异在算法侧容忍范围内。

当前进展：

- 已增加 `MODEL_CONCURRENCY_LIMIT`、`MODEL_QUEUE_TIMEOUT_SECONDS`，支持模型级并发限制和队列超时。
- 模型配置可使用 `max_concurrency`、`queue_timeout_seconds` 或 `runtime.max_concurrency`、`runtime.queue_timeout_seconds` 覆盖默认值。
- FP16 ONNX 输入会根据 session 输入 dtype 自动 cast。
- 已支持 `runtime: tensorrt` + `ENABLE_TENSORRT=true` 时优先使用 ONNX Runtime `TensorrtExecutionProvider`，并支持 TensorRT engine cache 配置。
- 已新增 `tools/worker_control.py`，支持对多个 worker 统一执行 health、ready、reload-config、aliases、warmup、reload 和 unload。
- 当前 `/vision/infer` 已支持单请求内 batch；跨请求动态 batch/批处理窗口仍作为后续可选优化，需结合真实 QPS 和延迟目标单独压测。

## 15. 推荐优先级

建议先做：

1. 模块化拆分。
2. 模型包契约。
3. 模型配置 schema。
4. 分类插件和通用 `/vision/infer`。
5. 模型上线前 smoke test。

暂缓做：

- 自动训练触发。
- 训练服务接入。
- 大规模实验管理。
- 跨请求动态 batch。

## 16. 交付清单

推理服务侧每次升级应交付：

- 代码变更。
- API 文档。
- `models.yml` 示例。
- 模型包契约说明。
- 测试用例。
- Docker 构建验证。
- 最小模型 smoke test。
- 上线/回滚操作说明。

## 17. 与算法研发方案的硬性对齐项

算法侧每个模型必须提供：

- ONNX 模型文件。
- 模型卡。
- labels 文件或类别定义。
- 输入输出契约。
- 样例图片和期望输出。
- 算法评估报告。
- 导出命令和环境。
- sha256。
- 推荐阈值。
- 线上限制说明，例如最大 batch、输入尺寸、是否支持动态 batch。

推理服务侧必须做到：

- 拒绝不完整或不合法的模型包。
- 使用模型卡和配置驱动预处理/后处理。
- 记录线上模型版本。
- 支持同一任务多版本共存。
- 支持快速回滚。
