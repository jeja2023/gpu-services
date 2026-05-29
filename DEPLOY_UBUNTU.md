# Ubuntu 服务器部署教程

本文档用于把本项目部署到 Ubuntu 服务器，通过 Docker Compose 启动两个 GPU 推理 worker，为其它业务容器提供 ONNX GPU 推理服务。

官方参考：

- Docker Engine Ubuntu 安装文档: <https://docs.docker.com/engine/install/ubuntu/>
- NVIDIA Container Toolkit 安装文档: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

## 1. 服务器准备

推荐环境：

- Ubuntu 22.04 LTS 或 24.04 LTS
- NVIDIA GPU 2 张，例如 2080 Ti
- NVIDIA 驱动正常
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- 服务容器内运行 Python 3.12，镜像构建时会安装 Python 3.12

先登录服务器：

```bash
ssh your_user@your_server_ip
```

更新系统包索引：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release git
```

## 2. 安装或确认 NVIDIA 驱动

如果服务器已经装好驱动，直接检查：

```bash
nvidia-smi
```

能看到 GPU 列表、驱动版本、显存信息，就可以继续。

如果没有驱动，可以用 Ubuntu 推荐驱动安装：

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

重启后再次确认：

```bash
nvidia-smi
```

## 3. 安装 Docker Engine 和 Compose 插件

卸载可能存在的旧版本：

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg"
done
```

添加 Docker 官方 apt 源：

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

安装 Docker：

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

验证：

```bash
docker --version
docker compose version
sudo docker run --rm hello-world
```

可选：允许当前用户免 `sudo` 运行 Docker：

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

## 4. 安装 NVIDIA Container Toolkit

添加 NVIDIA 官方 apt 源：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
```

安装：

```bash
sudo apt-get install -y nvidia-container-toolkit
```

配置 Docker runtime 并重启 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 容器：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

能在容器内看到 GPU 列表，说明 Docker GPU 环境正常。

## 5. 上传项目代码

方式一：使用 Git：

```bash
cd /opt
sudo mkdir -p /opt/gpu-services
sudo chown -R "$USER":"$USER" /opt/gpu-services
git clone <your_repo_url> /opt/gpu-services
cd /opt/gpu-services
```

方式二：从本地机器上传项目目录。下面命令在你的本地电脑执行，按实际本地路径调整：

```bash
scp -r /path/to/gpu-services your_user@your_server_ip:/opt/gpu-services
ssh your_user@your_server_ip
cd /opt/gpu-services
```

确认文件：

```bash
ls -la
```

应该能看到：

```text
Dockerfile
docker-compose.yml
main.py
requirements.txt
README.md
DEPLOY_UBUNTU.md
.env.example
```

## 6. 配置环境变量

复制配置模板：

```bash
cp .env.example .env
```

编辑：

```bash
nano .env
```

常用配置：

```dotenv
LOG_LEVEL=INFO
MAX_TENSOR_ITEMS=12582912
MAX_LOADED_MODELS=0
GPU_QUEUE_LIMIT=1
WARMUP_MODELS=
API_TOKEN=change-me-to-a-long-random-token
GPU_WORKER_0_DEVICE=0
GPU_WORKER_1_DEVICE=1
```

说明：

- `API_TOKEN` 建议生产环境设置为长随机字符串。
- `WARMUP_MODELS` 可写成 `person_service/reid.onnx,person_service/face.onnx`。
- 如果服务器只有 1 张 GPU，先删除或注释 `docker-compose.yml` 里的 `gpu-worker-1` 服务，或者只启动 `gpu-worker-0`。

生成随机 token 示例：

```bash
openssl rand -hex 32
```

## 7. 创建共享模型卷

创建 Docker volume：

```bash
docker volume create gpu-share-volume
```

查看卷位置：

```bash
docker volume inspect gpu-share-volume
```

准备本地模型目录，例如：

```bash
mkdir -p /opt/model-upload/person_service
cp /path/to/your_model.onnx /opt/model-upload/person_service/
```

把模型复制到共享卷，目录必须是 `项目名/models/模型文件`：

```bash
docker run --rm \
  -v gpu-share-volume:/projects \
  -v /opt/model-upload/person_service:/src:ro \
  ubuntu:22.04 \
  bash -lc "mkdir -p /projects/person_service/models && cp /src/*.onnx /projects/person_service/models/"
```

检查卷内模型：

```bash
docker run --rm -v gpu-share-volume:/projects ubuntu:22.04 \
  find /projects -maxdepth 4 -type f -name '*.onnx' -print
```

预期类似：

```text
/projects/person_service/models/your_model.onnx
```

## 8. 构建并启动服务

在项目目录执行：

```bash
cd /opt/gpu-services
docker compose up -d --build
```

构建镜像时会访问 Docker Hub、Ubuntu apt 源、deadsnakes Python 3.12 PPA、`bootstrap.pypa.io` 和 Python 包镜像源。服务器如果不能访问外网，建议在能联网的机器上构建镜像后推送到内网镜像仓库。

查看容器：

```bash
docker compose ps
```

查看日志：

```bash
docker compose logs -f --tail=100
```

只看某个 worker：

```bash
docker compose logs -f --tail=100 gpu-worker-0
```

确认容器内 Python 版本：

```bash
docker exec gpu-worker-0 python --version
```

预期为 Python 3.12.x。

## 9. 验证服务

检查健康状态：

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/ready
```

如果启用了 `API_TOKEN`：

```bash
TOKEN="你的API_TOKEN"
```

查看模型元信息：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:9001/model-info?project_name=person_service&model_name=your_model.onnx"
```

手动预热：

```bash
curl -X POST http://127.0.0.1:9001/warmup \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"models":[{"project_name":"person_service","model_name":"your_model.onnx"}]}'
```

查看 Prometheus 指标：

```bash
curl http://127.0.0.1:9001/metrics
```

推理请求示例：

```bash
curl -X POST http://127.0.0.1:9001/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Request-ID: test-001" \
  -H "Content-Type: application/json" \
  -d '{"project_name":"person_service","model_name":"your_model.onnx","tensor_data":[[[[0.1,0.2,0.3]]]]}'
```

注意：上面的 `tensor_data` 只是格式示例，实际 shape 必须匹配 ONNX 模型输入。

## 10. 业务容器接入

如果业务项目也运行在同一台 Docker 主机上，把业务容器加入 `gpu-bridge` 网络。

业务项目的 Compose 中加入：

```yaml
networks:
  gpu-bridge:
    external: true
```

服务中引用：

```yaml
services:
  your-business-service:
    networks:
      - gpu-bridge

networks:
  gpu-bridge:
    external: true
```

业务容器内调用：

```text
http://gpu-worker-0:8000/predict
http://gpu-worker-1:8000/predict
```

建议：

- 低延迟场景固定访问一个 worker，避免重复冷加载。
- 高吞吐场景在业务侧按 worker 做负载均衡。
- 调用方读取超时要覆盖排队时间、模型加载时间和推理时间。
- 生产环境建议统一携带 `X-Request-ID`，方便串联业务日志和推理日志。

## 11. 更新模型

把新 ONNX 文件复制进共享卷后，已加载模型不会自动热更新。

复制新模型：

```bash
docker run --rm \
  -v gpu-share-volume:/projects \
  -v /opt/model-upload/person_service:/src:ro \
  ubuntu:22.04 \
  bash -lc "cp /src/your_model.onnx /projects/person_service/models/your_model.onnx"
```

重载单个 worker 的模型：

```bash
curl -X POST http://127.0.0.1:9001/reload \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"project_name":"person_service","model_name":"your_model.onnx"}'
```

也可以直接重启 worker：

```bash
docker compose restart gpu-worker-0
docker compose restart gpu-worker-1
```

## 12. 常用运维命令

停止服务：

```bash
docker compose down
```

重启服务：

```bash
docker compose restart
```

重新构建：

```bash
docker compose up -d --build
```

查看资源：

```bash
docker stats
nvidia-smi
watch -n 1 nvidia-smi
```

进入容器：

```bash
docker exec -it gpu-worker-0 bash
```

容器内确认 ONNX Runtime provider：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 13. 常见问题

### `/ready` 返回没有 `CUDAExecutionProvider`

检查：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
docker compose logs --tail=100 gpu-worker-0
```

通常原因：

- 宿主机驱动未安装或异常。
- NVIDIA Container Toolkit 未安装。
- 没有执行 `sudo nvidia-ctk runtime configure --runtime=docker`。
- Docker 没有重启。

### `model not found`

确认模型路径：

```bash
docker run --rm -v gpu-share-volume:/projects ubuntu:22.04 \
  find /projects -maxdepth 4 -type f -print
```

服务要求路径：

```text
/projects/<project_name>/models/<model_name>
```

请求里的 `project_name` 和 `model_name` 必须和卷内目录、文件名完全一致。

### 显存不足

处理方式：

- 降低 batch size。
- 减少同一个 worker 加载的模型数。
- 设置 `MAX_LOADED_MODELS` 启用 LRU 缓存淘汰。
- 固定某些模型只走某张 GPU。
- 使用 FP16 / TensorRT 优化后的模型。

### 首次请求很慢

首次请求会加载 ONNX 模型并初始化 CUDA provider。可以使用：

- `WARMUP_MODELS` 启动预热。
- `/warmup` 手动预热。

### 外部机器访问不到 9001/9002

Compose 默认绑定：

```yaml
127.0.0.1:9001:8000
127.0.0.1:9002:8000
```

这是为了避免推理服务直接暴露公网。跨机器访问建议使用内网反向代理或 API 网关，并开启鉴权、限流和请求体大小限制。

## 14. 上线检查清单

- `nvidia-smi` 正常。
- GPU 容器内 `nvidia-smi` 正常。
- `docker compose ps` 显示 worker healthy。
- `/ready` 返回 `CUDAExecutionProvider`。
- `/model-info` 能返回正确输入 shape 和 dtype。
- `/warmup` 成功。
- 业务请求 tensor shape 与模型输入一致。
- 已设置 `API_TOKEN`。
- 已设置调用方超时、重试和日志 request id。
- 已完成至少一次真实模型压测。
