# VIDA Commercial API Runner

1. `git clone` 本仓库
2. 下载图片数据到本仓库下
  ```bash
  cd api_run

  hf download p1k0/visually-dependent-ambiguity \
  --repo-type dataset \
  raw/images.zip \
  raw/vida_sent.zip \
  --local-dir .

  unzip images.zip
  unzip vida_sent.zip
  ```

3. 修改`api_key.txt`。第一行是api，第二行是base_url
4. 运行`bash start.sh`
5. 结果在`output`，打包文件夹