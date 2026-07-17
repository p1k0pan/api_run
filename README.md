# 2026.7.17
调用qwen3.7-max语言模型来跑数据。两份数据分别跑：
1. `herbarium/au_stage1_batch_requests_pathcount1.jsonl`
2. `herbarium/de_stage1_batch_requests_pathcount1.jsonl`
   
需要注意除了`messages`字段我不确定其他字段是否是这样写的，**请你帮我确认一下**。现在的输入数据字段是这样的：
```json
{
  "custom_id": "node_au_...",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "qwen3.7-max",
    "enable_thinking": true,
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "response_format": {"type": "json_object"}
  }
}
```
或许可以先跑一条试试水😂


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