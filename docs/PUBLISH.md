# 公开代码与介绍页

## 通过 GitHub 网页上传

1. 登录 GitHub，创建一个 Public 仓库，例如 `wbc-rk3568`。描述可填：`RK3568 端侧白细胞检测、分类与样本上传模块`。
2. 在空仓库页面选择上传现有文件，将代码包解压后的文件和目录拖入上传区。保留 `src/`、`config/`、`docs/` 等结构，不要上传 ZIP 代替目录。
3. 提交后，代码访问网址为 `https://github.com/账号名/wbc-rk3568`。访客可以查看代码或下载 ZIP。
4. 如需介绍网页，进入仓库 Settings → Pages，选择 Deploy from a branch，分支选 `main`，目录选 `/docs`，保存。
5. Pages 完成发布后，访问 `https://账号名.github.io/wbc-rk3568/`。仓库 Settings → Pages 会显示实际地址。

介绍页展示模块功能与数据流；GitHub Pages 提供静态网页，不运行 RKNN，也不接收样本。

## 通过命令行推送

在代码包目录运行以下命令。将示例用户名、邮箱和 URL 替换为自己的值。可以在 GitHub 邮箱设置中取得 noreply 邮箱，避免公开常用邮箱；公开仓库仍会显示仓库所属账号和提交账号。

```bash
git init -b main
git config user.name "your-account-name"
git config user.email "your-github-noreply-address"
git add .
git commit -m "Add RK3568 edge image processing modules"
git remote add origin https://github.com/账号名/wbc-rk3568.git
git push -u origin main
```

GitHub HTTPS 推送使用 Git Credential Manager、`gh auth login` 或 personal access token，不使用账号密码。登录过程在自己的电脑完成，不把 token 写入配置或代码。

## 通过阿里云提供代码下载和介绍页

也可以将 `docs/index.html` 和代码 ZIP 放到自己的静态网站或 OSS 静态站点。将 ZIP 文件链接加入页面后，访客即可下载。配置公网访问时，仅公开代码及介绍页，不将整个开发目录作为网站根目录。

板端推理运行在 RK3568 上；如需远程看结果或接收上传，另外部署带身份验证的服务端应用，并将其 HTTPS 地址填入 `upload.url`。

官方操作参考：[创建 GitHub 仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)、[配置 GitHub Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
