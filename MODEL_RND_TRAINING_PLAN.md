# 训练、标注与模型研发配套方案

本文档面向独立于 `gpu-services` 的算法研发体系。目标是建立一套可持续的图像识别模型研发、标注、训练、评估、导出和交付流程，并与 `INFERENCE_SERVICE_UPGRADE_PLAN.md` 中的推理服务升级方案匹配。

核心原则：训练研发侧不直接侵入推理服务代码，推理服务也不承载训练任务。双方通过标准模型包、模型卡、配置契约和验收样例对接。

## 1. 目标定位

算法研发体系负责：

- 数据采集和数据版本管理。
- 标注规范、标注质检和标注集维护。
- 模型训练、微调、蒸馏、量化和实验管理。
- 离线评估、误差分析和模型选择。
- 模型导出为 ONNX。
- 生成标准模型包并交付给推理服务。

算法研发体系不负责：

- 线上 API 服务。
- 线上模型缓存、灰度和回滚。
- 线上请求鉴权。
- 线上视频流读取。
- 业务系统接口适配。

这些由 `gpu-services` 推理服务负责。

## 2. 与推理服务的匹配边界

算法侧的最终产物不是“训练好的权重文件”这么简单，而是可上线的标准模型包。

推理服务只承诺消费以下内容：

- 模型文件：优先 ONNX。
- 模型配置：输入输出、任务类型、后处理参数。
- 模型卡：版本、指标、适用场景和限制。
- labels：类别或标签定义。
- 样例数据：用于 smoke test。
- 验收指标：用于服务侧上线验证。

算法侧不应要求推理服务理解训练框架内部细节，例如 PyTorch checkpoint 的网络结构、训练脚本参数、数据增强策略等。这些可以写入模型卡，但不作为服务运行时依赖。

## 3. 推荐仓库结构

建议单独建立算法研发仓库，例如：

```text
vision-model-lab/
  README.md
  configs/
    datasets/
    experiments/
    export/
  data/
    README.md
    manifests/
    splits/
  labeling/
    guidelines/
    quality_rules/
    review_reports/
  src/
    datasets/
    models/
    training/
    evaluation/
    export/
    packaging/
  notebooks/
  scripts/
    prepare_dataset.py
    train.py
    evaluate.py
    export_onnx.py
    package_model.py
  experiments/
    local_runs/
  artifacts/
    packages/
  tests/
```

大型图片、视频、权重和数据集不建议直接提交 Git。推荐使用：

- NAS/对象存储/MinIO/S3 保存原始数据和模型制品。
- Git 保存配置、脚本、标注规范、manifest、模型卡模板。
- DVC、MLflow、Weights & Biases 或自建表记录实验版本。

## 4. 任务类型规划

第一批建议支持和推理服务匹配的任务：

| 任务 | 训练侧产物 | 推理侧插件 |
| --- | --- | --- |
| 图片分类 | 分类 ONNX、labels、top-k 规则 | `classification` |
| 目标检测 | YOLO/RT-DETR 等 ONNX、类别、阈值 | `yolo_detection` |
| 人体 ReID | embedding ONNX、归一化方式 | `reid` |
| 图像分割 | segmentation ONNX、mask 后处理规则 | `segmentation` |
| OCR | 检测/识别 ONNX、字典、解码规则 | `ocr` |
| 多阶段流水线 | 多个模型包和组合规则 | `multi_stage_pipeline` |

建议优先顺序：

1. 检测和 ReID：延续当前项目已有能力。
2. 分类：实现成本低，适合作为插件化的第一块验证。
3. 分割：输出结构复杂，用于验证模型包契约是否足够。
4. OCR 和多阶段流水线：放在契约稳定之后。

## 5. 数据管理方案

数据应分层：

```text
datasets/
  raw/
    project_a/
  labeled/
    project_a/
  curated/
    project_a/
  evaluation/
    project_a/
  manifests/
    project_a_train_v1.jsonl
    project_a_val_v1.jsonl
    project_a_test_v1.jsonl
```

各层含义：

- `raw`：原始数据，不直接训练。
- `labeled`：完成标注的数据。
- `curated`：清洗、去重、质检后的训练数据。
- `evaluation`：固定评估集，不随训练随意变化。
- `manifests`：训练、验证、测试划分清单。

manifest 建议使用 JSONL：

```json
{"image":"s3://bucket/project/frame001.jpg","label":"s3://bucket/project/frame001.json","split":"train","source":"camera_01","tags":["day","indoor"],"dataset_version":"1.0.0"}
```

数据版本规则：

- `major`：标注定义或任务定义改变。
- `minor`：新增一批数据或明显改进质检。
- `patch`：修复少量错误标注。

示例：

```text
person_detection_dataset_v1.2.0
product_defect_classification_dataset_v2.0.1
```

## 6. 标注方案

### 6.1 标注规范

每个任务必须有独立标注规范：

- 类别定义。
- 正例和反例。
- 困难样本规则。
- 遮挡、模糊、截断的处理。
- 多目标重叠规则。
- 小目标是否标注。
- 忽略区域定义。
- 不确定样本如何处理。

检测任务示例：

- bbox 必须覆盖完整目标可见区域。
- 被严重遮挡但仍可识别时标注可见区域。
- 小于指定像素阈值的目标可标为 ignore。
- 类别冲突时进入复审池。

分类任务示例：

- 每张图可以单标签或多标签，必须提前定义。
- 模糊图、曝光异常图是否参与训练必须明确。
- 不确定标签进入 `uncertain`，不直接作为训练标签。

### 6.2 标注质检

质检流程：

1. 首轮标注。
2. 抽样复审。
3. 高风险类别全量复审。
4. 模型辅助发现异常标注。
5. 生成质检报告。

质检指标：

- 标注通过率。
- 类别混淆样本数。
- bbox 平均偏差。
- 漏标率。
- 重复标注率。
- `uncertain` 样本比例。

建议规则：

- 新标注员前 500 张全量复审。
- 稳定标注员抽样复审比例不低于 10%。
- 线上高风险任务的关键类别复审比例不低于 30%。
- 每次数据集版本升级必须产出质检报告。

## 7. 训练流程

训练流水线：

1. 选择任务和数据集版本。
2. 固定训练/验证/测试划分。
3. 确定 baseline 模型。
4. 训练模型。
5. 保存 checkpoint。
6. 在验证集上选最优模型。
7. 在测试集和业务回归集上评估。
8. 做误差分析。
9. 导出 ONNX。
10. 服务侧兼容性预验证。
11. 打包标准模型包。

训练配置必须记录：

- 代码版本。
- 数据集版本。
- 模型结构。
- 预训练权重。
- 输入尺寸。
- batch size。
- epoch。
- optimizer。
- learning rate。
- augmentation。
- random seed。
- 硬件环境。
- 训练耗时。

建议每次训练生成：

```text
experiments/
  person_detector/
    2026-06-03_yolov8n_v1/
      config.yml
      train.log
      metrics.json
      confusion_matrix.png
      checkpoints/
      eval/
      export/
```

## 8. 评估和验收

不同任务使用不同指标：

| 任务 | 核心指标 | 辅助指标 |
| --- | --- | --- |
| 分类 | accuracy、precision、recall、F1、AUC | top-k、混淆矩阵 |
| 检测 | mAP、AP50、AP75、precision、recall | 小目标 AP、漏检率、误检率 |
| 分割 | mIoU、Dice、mask AP | 边界误差、面积误差 |
| ReID | mAP、Rank-1、Rank-5 | embedding 分布、类内/类间距离 |
| OCR | 字符准确率、文本准确率 | 检测召回、编辑距离 |

每个模型必须在以下集合上评估：

- 训练集：只看是否欠拟合。
- 验证集：用于选模型。
- 固定测试集：用于版本比较。
- 业务回归集：用于上线前把关。
- 边界样本集：用于观察异常场景。

上线建议门槛：

- 指标不低于当前线上版本。
- 如果局部指标下降，必须明确业务可接受原因。
- 推理延迟满足服务侧要求。
- ONNX 输出与训练框架输出差异在容忍范围内。
- FP16/INT8 版本必须和 FP32 版本对比。

## 9. ONNX 导出规范

主部署格式为 ONNX。

推荐输入：

```text
layout: NCHW
shape: [batch, 3, height, width]
dtype: float32 或 float16
color: RGB
```

建议：

- 高优先级模型保留 FP32 基准版。
- 生产可增加 FP16 版本。
- INT8 只有在校准集和精度评估完整时使用。
- 输入 H/W 尽量固定，batch 可按服务需要选择动态或固定。
- 导出后必须运行 ONNX Runtime 校验。
- 输出 tensor 的语义必须写入模型卡。

导出后校验：

- 模型能被 `onnx.checker` 校验。
- ONNX Runtime CPU 可运行。
- ONNX Runtime GPU 可运行。
- 与 PyTorch 原模型输出差异在阈值内。
- 多张样例图片输出稳定。
- 服务侧预处理参数与训练/导出参数一致。

## 10. 标准模型包

模型包必须能被推理服务直接消费。

短期兼容当前 `gpu-services` 路径：

```text
shared-models/
  <project_name>/
    <artifact_name>.onnx
    <artifact_name>.model-card.yml
    <artifact_name>.labels.txt
    <artifact_name>.examples/
      input_001.jpg
      expected_001.json
```

示例：

```text
shared-models/
  cross_camera_tracking/
    person_detector_yolov8n_v1.0.0_fp32.onnx
    person_detector_yolov8n_v1.0.0_fp32.model-card.yml
    person_detector_yolov8n_v1.0.0_fp32.labels.txt
    person_detector_yolov8n_v1.0.0_fp32.examples/
      frame_001.jpg
      frame_001.expected.json
```

模型包命名：

```text
<task>_<architecture>_v<semver>_<precision>.onnx
```

示例：

```text
person_detector_yolov8n_v1.0.0_fp32.onnx
person_reid_osnet_ibn_x1_0_v1.1.0_fp32.onnx
defect_classifier_resnet50_v2.0.0_fp16.onnx
```

模型卡示例：

```yaml
model:
  name: person_detector_yolov8n
  version: 1.0.0
  task: detection
  architecture: yolov8n
  precision: fp32
  format: onnx
  sha256: ""

dataset:
  train: person_detection_dataset_v1.2.0
  val: person_detection_dataset_v1.2.0
  test: person_detection_test_v1.0.0

input:
  layout: nchw
  shape: [1, 3, 640, 640]
  dtype: float32
  color: rgb
  resize: letterbox
  normalize: none

output:
  format: yolo
  classes: coco
  recommended_confidence: 0.25
  recommended_iou: 0.45

metrics:
  map50: 0.0
  precision: 0.0
  recall: 0.0
  latency_ms:
    gpu: 0.0

deployment:
  runtime: onnxruntime
  min_gpu_memory_mb: 0
  max_batch_size: 16
  supports_dynamic_batch: true

limitations:
  - "低照度场景需要继续补充数据。"
```

## 11. 与 `models.yml` 的对齐

算法侧交付模型包时，应同时提供建议配置片段：

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
      model_card: person_detector_yolov8n_v1.0.0_fp32.model-card.yml
      labels: person_detector_yolov8n_v1.0.0_fp32.labels.txt
```

推理服务最终以 `models.yml` 为运行配置，以模型卡为审计和验收材料。两者不一致时，模型包应进入待确认状态，不能直接上线。

算法侧交付前应使用推理服务侧校验脚本做一次自检。假设 `gpu-services` 与 `shared-models` 同级：

```bash
cd gpu-services
python tools/validate_model_package.py \
  --config models.yml \
  --models-root ../shared-models \
  --strict-hash \
  --strict-sidecars \
  --model-id cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx
```

自检通过后再进入服务侧 smoke test、预热、灰度和回滚流程。未通过时应优先补齐模型卡、labels 或类别定义、sha256、输入输出契约和推荐阈值。

如果模型包包含样例图片和期望输出，建议同时交付 `regression.yml`：

```yaml
tolerance: 0.001
cases:
  - name: detector_frame_001
    method: POST
    path: /vision/infer
    form:
      model_id: person_detector_candidate
      confidence: "0.25"
      iou: "0.45"
    files:
      files: person_detector_yolov8n_v1.0.0_fp32.examples/frame_001.jpg
    expected_path: person_detector_yolov8n_v1.0.0_fp32.examples/frame_001.expected.json
```

推理服务侧可用 `tools/regression_check.py` 执行固定样例回归，作为候选模型切换到 active alias 前的门禁。
如果走按权重灰度，回归样例或业务灰度请求应传入稳定的 `traffic_key`，例如客户 ID、设备 ID 或场景 ID，确保同一个 key 在灰度期间稳定命中同一模型版本。

## 12. 实验管理

建议每次实验记录：

- experiment_id。
- git commit。
- 数据集版本。
- 配置文件。
- 训练日志。
- 评估指标。
- 导出结果。
- 是否进入候选模型。
- 与线上模型对比结论。

可以使用：

- MLflow：适合自建和制品记录。
- Weights & Biases：适合团队可视化和实验追踪。
- DVC：适合数据版本。
- 简化阶段也可以先用 YAML/JSON + 目录归档。

最低可行版本：

```text
experiments/index.yml
artifacts/packages/
```

`index.yml` 示例：

```yaml
experiments:
  - id: person_detector_20260603_001
    task: detection
    dataset: person_detection_dataset_v1.2.0
    model: yolov8n
    status: packaged
    package: cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx
```

## 13. 误差分析闭环

线上推理服务应把问题样本或统计信息反馈给算法侧。算法侧建立问题池：

```text
error_analysis/
  false_positive/
  false_negative/
  low_confidence/
  domain_shift/
  bad_image_quality/
  annotation_issue/
```

闭环流程：

1. 线上发现问题。
2. 业务侧或服务侧保存样本和模型版本。
3. 算法侧归因。
4. 若是标注问题，进入标注修复。
5. 若是数据不足，进入数据补采。
6. 若是模型能力不足，进入新实验。
7. 新版本模型包交付给推理服务。
8. 推理服务灰度上线。

每个线上问题样本必须记录：

- 图片或视频帧地址。
- 请求时间。
- 模型版本。
- 输入尺寸。
- 输出结果。
- 人工判定。
- 错误类型。

## 14. 角色分工

算法工程师：

- 训练模型。
- 评估指标。
- 导出 ONNX。
- 编写模型卡。
- 提供样例和期望输出。

标注负责人：

- 维护标注规范。
- 管理标注任务。
- 输出质检报告。
- 维护数据集版本。

推理服务工程师：

- 维护 `gpu-services`。
- 实现任务插件。
- 模型包校验。
- API、监控、灰度和回滚。

业务负责人：

- 定义业务目标。
- 确认误检/漏检成本。
- 接受或拒绝模型指标变化。
- 参与灰度验收。

## 15. 分阶段实施计划

### 第 0 阶段：建立基础规则

目标：先把交付物说清楚。

工作：

- 建立模型命名规则。
- 建立模型卡模板。
- 建立数据集版本规则。
- 建立标注规范模板。
- 建立模型包目录结构。
- 为当前 YOLO 和 ReID 模型补齐模型卡。

验收：

- 当前模型能被描述为标准模型包。
- 推理服务可依据模型包信息补充 `models.yml`。

### 第 1 阶段：建立训练和导出流水线

目标：让训练结果可以稳定变成 ONNX。

工作：

- 固化训练配置。
- 固化评估脚本。
- 固化 ONNX 导出脚本。
- 增加 ONNX Runtime 对比测试。
- 增加模型包打包脚本。

验收：

- 任意候选模型都能生成 ONNX、模型卡、labels 和样例。
- 导出的 ONNX 能在本地 ONNX Runtime 跑通。

### 第 2 阶段：建立固定评估集和回归集

目标：避免模型上线只看单次训练指标。

工作：

- 建立固定测试集。
- 建立业务回归集。
- 建立边界样本集。
- 输出版本对比报告。
- 为推理服务 smoke test 提供最小样例。

验收：

- 每个模型版本都能和线上版本对比。
- 模型上线前能给出明确建议：上线、灰度、拒绝。

### 第 3 阶段：接入推理服务上线流程

目标：模型包能进入推理服务候选、预热、灰度、回滚流程。

工作：

- 把模型包放入 `shared-models` 或制品仓库。
- 提供 `models.yml` 建议配置。
- 推理服务执行模型包校验。
- 推理服务执行 smoke test。
- 业务侧执行灰度验收。

验收：

- 新模型不需要改推理服务核心代码即可上线。
- 上线失败可以回滚旧版本。

### 第 4 阶段：性能和量化

目标：在保证准确率的前提下提升推理性能。

工作：

- FP16 导出和评估。
- INT8 校准集建设。
- TensorRT 构建测试。
- 与 FP32 基准对比。
- 输出性能和准确率折中报告。

验收：

- 每个加速版本都有准确率对比。
- 服务侧延迟收益明确。
- 不达标版本不进入生产。

## 16. 交付清单

每个候选模型版本必须交付：

- `<artifact_name>.onnx`
- `<artifact_name>.model-card.yml`
- `<artifact_name>.labels.txt`
- `<artifact_name>.examples/`
- `expected_*.json`
- 评估报告。
- 导出日志。
- sha256。
- `models.yml` 建议片段。
- 与上一线上版本的对比结论。

## 17. 上线判定模板

上线前给出如下结论：

```yaml
decision:
  model: cross_camera_tracking/person_detector_yolov8n_v1.0.0_fp32.onnx
  recommendation: gray_release
  reason:
    - "固定测试集 mAP50 高于线上版本。"
    - "低照度样本召回略低，建议先灰度。"
  required_service_checks:
    - smoke_test
    - model_package_check
    - latency_check
  rollback_target: cross_camera_tracking/person_detector_yolov8n_v0.9.0_fp32.onnx
```

可选结论：

- `reject`：拒绝上线。
- `lab_only`：仅实验室使用。
- `gray_release`：允许灰度。
- `production`：允许全量上线。

## 18. 与推理服务方案的硬性对齐项

算法侧必须遵守：

- 主部署格式为 ONNX。
- 模型包命名包含任务、架构、版本和精度。
- 模型卡中的输入输出契约必须真实可复现。
- 每个模型必须有样例和期望输出。
- 每个上线模型必须有对比报告。

推理服务侧必须遵守：

- 不依赖训练框架 checkpoint。
- 不猜测模型预处理和后处理。
- 不上线缺少模型卡和样例的模型。
- 不静默切换模型版本。
- 线上日志必须记录模型版本。
