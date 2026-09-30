# 模型文件

将检测模型保存为 `wbc_yolov5.rknn`，分类模型保存为 `wbc_classifier.rknn`。
两个文件均需以 RK3568 为目标平台导出，并与板端 RKNN Runtime、NPU 驱动兼容。
模型权重不随代码分发。

检测模型支持两种输出，详见 `docs/MODEL_CONTRACT.md`。分类模型输出五分类向量。
输入为 NHWC RGB uint8；归一化应在 RKNN 转换配置中完成。属性分支不属于部署输出。
