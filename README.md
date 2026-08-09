# Anime Upscale Web

一个面向单机、单 GPU 的轻量视频超分任务界面。当前生产模型固定为
StarSample V2 Lite，浏览器负责创建、取消和重试任务，容器内的单个 Worker
依次完成 FFmpeg 解码、逐帧超分、HEVC Main10 编码和音字幕封装。

## 前置条件

- Docker Engine 与 Docker Compose 插件
- NVIDIA 驱动及 NVIDIA Container Toolkit
- Docker 能通过 `gpus: all` 访问本机 GPU
- 本机媒体位于 `/mnt/user/data`

## 模型权重

将模型放到：

```text
models/2x-StarSample-V2-Lite.safetensors
```

预期 SHA256：

```text
4008dfc72295bb48574a389bf4bd4e55d9af3766f34b6b68cc7bc0c78bd22a0b
```

可在启动前校验：

```bash
sha256sum models/2x-StarSample-V2-Lite.safetensors
```

模型权重不会提交到 Git；`.gitignore` 已忽略整个 `models/` 目录。发布到
GitHub 时只提交代码，使用者需要自行提供并校验权重。

## 启动

国内网络环境建议在首次构建时使用清华 PyPI 镜像：

```bash
docker compose -f compose.web.yaml build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

镜像地址只影响 Docker 构建过程，不会在宿主机安装 Python 包。也可以省略
`--build-arg`，此时使用官方 PyPI。

```bash
mkdir -p output data
docker compose -f compose.web.yaml up -d
```

浏览器打开：

```text
http://<服务器 IP>:8787
```

查看日志或停止服务：

```bash
docker compose -f compose.web.yaml logs -f
docker compose -f compose.web.yaml down
```

## 挂载与输出

- `/mnt/user/data` 以相同绝对路径只读挂载，兼容目录中的符号链接。
- `./output` 挂载到 `/output`，存放完成的视频。
- `./data` 挂载到 `/data`，持久保存 SQLite 队列和任务日志。
- `./models` 只读挂载到 `/models`。

Compose 会向容器暴露 NVIDIA 的 `compute`、`utility` 和 `video` 能力，以便
PyTorch CUDA 推理和 NVENC 编码同时可用。

## 当前行为

- Uvicorn 固定为一个进程，Tesla P4 同一时间只运行一个超分任务。
- 支持选择单个视频或递归加入一个目录，输入文件不会被修改。
- 输出固定写到 `/output`，已有目标不会覆盖，任务失败不会冒充为完成文件。
- 音轨、字幕、章节和附件会尽可能从源文件直接复制，默认 MKV 输出最适合
  保留 ASS/PGS 字幕和字体附件。
- 可以取消排队中或运行中的任务，并可以重试失败、取消的任务。
- 队列和日志会跨容器重启保留。运行中重启的任务会标记失败，可从头重试；
  当前版本不提供逐帧断点续跑。
- 输入视频需要可识别的固定帧率。第一版不把可变帧率素材当作生产目标。

首次使用建议先提交一段短片，并在页面状态中确认 CUDA 可用。随后用
`ffprobe` 检查输出的分辨率、帧率、音轨、字幕和章节，再提交完整剧集。

此服务没有账号和权限系统，只应部署在可信局域网内，不要直接暴露到公网。
