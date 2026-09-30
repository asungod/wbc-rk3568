# 模型输入输出接口

两个模型均以 NHWC、RGB、uint8、batch=1 输入。检测输入默认 640×640，分类输入默认 224×224。检测采用灰色 letterbox，分类采用裁剪区域直接缩放。若训练使用中心裁剪或不同颜色处理，需相应调整预处理。

归一化在 ONNX 图或 RKNN 转换配置中完成，只处理一次。例如是否使用 ImageNet mean/std 必须依据训练设置，不能仅凭网络名称决定。

## 检测输出

`format = yolov5_heads_nchw`：三个浮点输出 `[1, A*(5+C), H, W]`，从高分辨率到低分辨率排列；每个 anchor 对应 `x,y,w,h,objectness,class_prob...`。标准三尺度常为 H/W=80、40、20。`anchors` 每组必须与对应输出一致。`head_logits=false` 表示图内已做 sigmoid，`true` 表示图外进行 sigmoid。双重 sigmoid 会导致错误结果。

`format = decoded_xywh`：一个浮点输出 `[1,N,5+C]` 或 `[N,5+C]`，坐标已解码为检测输入像素单位的中心 x、中心 y、宽、高，后续为 objectness 与各类别概率。不是归一化坐标，也不是 xyxy / NMS 后 `[N,6]` 格式。

只筛选 `wbc_class_id` 类，再按 objectness×类别概率阈值过滤并进行 NMS。整数输出必须先依照各张量量化 scale/zero-point 反量化；代码拒绝未知整数输出。

## 分类输出

单个五分类浮点向量，允许 `[1,5]` 等可展平为五元素的形状。`output=logits` 使用稳定 softmax，`output=probabilities` 检查概率范围及和为 1。`labels` 必须与训练环境的类别索引完全一致，可从训练导出记录或 `class_to_idx` 获取；文件名不能证明顺序。

## 板端接入检查

1. 记录板卡系统、Python、SDK、NPU 驱动及 Runtime 版本。
2. 确认模型目标平台为 RK3568。
3. 检查输出数量、尺寸、激活与量化形式，再选择配置。
4. 用同一张输入图像比较训练端与板端的候选框、类别和概率。
5. 检查相机断开、无目标、上传超时及重复点击的行为。

端侧模块消费部署模型，不承担 WBCAtt 属性预训练或 Raabin-WBC 微调。
