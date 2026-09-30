# RK3568 白细胞端侧分析

面向 RK3568 Linux 的白细胞图像处理模块，将显微图像输入、YOLOv5 候选区域检测、五分类推理及待复核样本上传串联为一个端侧流程。通过 RKNN-Toolkit Lite2 调用 NPU，检测与分类采用独立模型，属性训练分支不参与板端推理。

## 功能

- USB 摄像头采集与本地图像输入。
- 640×640 检测输入、letterbox 预处理、坐标还原及非极大值抑制。
- 候选区域筛选、边界裁剪和 224×224 五分类输入。
- 分类概率解析、低置信度标记、原图与标注图保存。
- Qt 图片预览、拍照分析、待复核样本手动上传。
- HTTP multipart 上传图片和 `result.json`，失败时保留本地文件。

## 环境与模型

板端使用 Linux aarch64、Python 3.9 或以上版本、可用的 NPU 驱动和匹配版本的 RKNN Runtime / RKNN-Toolkit Lite2。Lite2 的 wheel 请从 [Rockchip SDK](https://github.com/airockchip/rknn-toolkit2) 获取，按板端 Python 与系统版本选择，不直接安装 PC 转换工具替代它。

在代码目录运行：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
# 如需 Qt 界面，安装 GUI 依赖；也可按板端发行版安装兼容的 PySide6。
python -m pip install -e '.[gui]'
# 将与系统匹配的 Lite2 wheel 路径替换进下面的命令
python -m pip install /path/to/rknn_toolkit_lite2-compatible.whl
```

将两个 RKNN 文件放入 `models/`。编辑 `config/edge.json`，核对类别顺序、模型输出形式、anchors 和预处理配置，详见 [模型接口](docs/MODEL_CONTRACT.md)。配置中的类别顺序和 anchors 是可编辑的接口设置，必须与实际导出的模型一致。

```bash
wbc-edge --config config/edge.json --image /path/to/image.jpg
wbc-edge --config config/edge.json --camera 0
wbc-gui --config config/edge.json
```

结果位于 `output/<sample_id>/`：

```text
image.jpg       原始图像
annotated.jpg   候选框、类别及置信度
result.json     样本编号、坐标、概率、待复核标志和本次处理耗时
```

`analysis_ms` 是当前一次端侧处理测量值，不包括模型初始化、拍照、保存或上传时间。零候选结果保存为空列表，不生成分类结果。

## 网络上传

4G 模块通过操作系统网络接口提供连接；应用使用 HTTP(S)，不直接控制 SIM、串口拨号或网络注册。先配置系统联网，再在配置文件设置 `upload.url`。非本机地址使用 HTTPS。

```bash
export WBC_UPLOAD_TOKEN='your-service-token'
wbc-edge --image /path/to/image.jpg --upload
```

CLI 只在指定 `--upload` 且样本需要复核时发起上传。Qt 界面通过按钮手动上传。服务器需接收 `image` 和 `result` 两个 multipart 字段，成功返回 2xx；鉴权使用可选 Bearer token。`Idempotency-Key` 为样本编号，服务器应按该键去重。上传完成表示接收成功，不改变分类或复核状态。

## 目录

```text
src/wbc_edge/   端侧模块与 Qt 界面
config/        运行配置
models/        模型存放位置
tests/         CPU 功能测试
docs/          模型接口、发布步骤与介绍页
```

本仓库公开端侧应用模块。模型权重、训练数据、训练脚本、服务器业务系统不包含在代码包内。

## 检查与发布

```bash
python -m unittest discover -s tests -v
```

测试覆盖坐标映射、NMS、输出检查、资源释放、结果保存和上传协议。发布代码与公开介绍页的方法见 [发布说明](docs/PUBLISH.md)。

相关接口文档：[RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)、[Rockchip YOLOv5 部署例程](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov5)。
