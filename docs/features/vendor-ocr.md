# Vendor OCR

**Pitch:** Point a Baidu, Alibaba Cloud, or Tencent Cloud general-OCR client at this Mac. The same Unlimited OCR 2 pass that serves `/v1/chat/completions` answers, in that vendor's JSON shape.

百度通用文字识别、阿里云 RecognizeGeneral、腾讯云 GeneralBasicOCR / GeneralAccurateOCR 三种请求形状。背后仍是 `baidu/Unlimited-OCR-2` 的 `document parsing.`，不是第二套 OCR。`POST /v1/chat/completions` 保持原样。

## Who it is for

已经接好百度、阿里云或腾讯云通用 OCR、想把主机改到本机 NovaMLX 的人。身份证、银行卡、增值税发票这类专用接口不在这一版里。

## What it does

一条引擎路径，三套薄适配。图片（base64、http(s) URL、`data:` URL、multipart 文件，或路径以 `.pdf` 结尾的 PDF）进同一个 VLM 请求：user 文本固定为 `document parsing.`，模型默认 `baidu/Unlimited-OCR-2`。多页时处理器自己会把提示改成 `Multi page parsing.`。模型吐出的 `<|det|>`、`<|ref|>`、`<table>`、`<PAGE>` 被拆成文字行和框，再装进各厂商的成功 JSON。

`accurate` 与 `basic`、`RecognizeAdvanced` 与 `RecognizeGeneral`、`GeneralAccurateOCR` 与 `GeneralBasicOCR` 走同一次识别。路径只为了让旧客户端能换 host。

| 厂商 | 方法与路径 | 成功体 |
|------|------------|--------|
| 百度 | `POST /rest/2.0/ocr/v1/general_basic` | `log_id`、`words_result_num`、`words_result[].words`。有框时带 `location`（`left` / `top` / `width` / `height`） |
| 百度 | `POST /rest/2.0/ocr/v1/accurate_basic` | 同上 |
| 百度 | `POST /rest/2.0/ocr/v1/general` | 同上 |
| 百度 | `POST /rest/2.0/ocr/v1/accurate` | 同上 |
| 阿里云 | `POST /ocr/RecognizeGeneral` | `RequestId` + `Data`（JSON 字符串，内含 `content`、`prism_wnum`、`prism_wordsInfo`） |
| 阿里云 | `POST /ocr?Action=RecognizeGeneral` | 同上。`Action=RecognizeAdvanced` 也进这条 |
| 腾讯云 | `POST /ocr/GeneralBasicOCR` | `Response.TextDetections[]`：`DetectedText`、`Confidence`、`Polygon`、`ItemPolygon`、`AdvancedInfo` |
| 腾讯云 | `POST /ocr/GeneralAccurateOCR` | 同上 |
| 腾讯云 | `POST /ocr?Action=GeneralBasicOCR` | 同上。也认头 `X-TC-Action` 和 JSON 字段 `Action` |

专用产品（`/rest/2.0/ocr/v1/idcard`、`IDCardOCR`、`RecognizeIdcard`、银行卡、发票等）没有实现。落在 `/rest/2.0/ocr/` 或 `/ocr/` 下的未知路径返回该厂商的错误体，而不是通用识别的结果。

### 鉴权

与 `/v1/chat/completions` 相同：`Authorization: Bearer <Nova key>`，或 `x-api-key`。没配 key 时仍是开放模式。

这些厂商字段不是 Nova 的凭证，有 Bearer / `x-api-key` 时会被忽略：

- 百度 query 或表单里的 `access_token`
- 腾讯云 `SecretId` / `SecretKey`
- 阿里云 ACS、腾讯云 TC3 写在 `Authorization` 里的签名

只带厂商签名、不带 Nova key 时，中间件仍返回 OpenAI 形态的 401。把 Nova key 放在 `Bearer` 或 `x-api-key` 上。`Authorization` 若是 ACS / TC3 而不是 `Bearer`，可以同时带 `x-api-key`。

### 请求里的图

JSON、`application/x-www-form-urlencoded`、`multipart/form-data`、或原始 `image/*` / `application/pdf` 正文都可以。

| 字段 | 含义 |
|------|------|
| `image`、`ImageBase64`、`img`、`body` | base64，或 `data:` URL。已经是 `http(s)://` 时当作 URL |
| `url`、`Url`、`ImageUrl`、`image_url` | 图片 URL。与 base64 同时出现时用 base64 |
| `pdf_file` | PDF 的 base64。写成临时 `.pdf` 再走聊天接口已有的 PDF 渲染 |
| `model` 或头 `X-Nova-Model` | 换模型。缺省 `baidu/Unlimited-OCR-2`。头优先于 query，query 优先于正文 |

`language_type`、`detect_direction`、`paragraph`、`probability`、`LanguageType`、`Scene` 不参与识别。

框：`<|det|>` 里的坐标若都在 0…1000，且请求里带了图片字节（base64、multipart、原始正文），会按 999 映射到像素。大于 1000 的数当作像素，不再缩放。只有 URL、没有字节时，框留在模型坐标系里。模型不给置信度：腾讯云 `Confidence` 和阿里云 `prob` 固定为 `0`。

百度错误仍是 HTTP 200，看 `error_code` / `error_msg`（缺图 `216101`，格式 `216201`，未实现的专用接口 `3`，引擎失败 `282000`）。阿里云用 `Code` / `Message`，缺图 400，引擎失败 500，模型不可用 503。腾讯云 HTTP 200，错误在 `Response.Error`。

## How to try

先让聊天接口能跑（权重与加载见 [Unlimited OCR](unlimited-ocr.md)）。下面的例子假设服务在 `127.0.0.1:6590`。

聊天路径不变：

```bash
IMG=$(base64 -i page.png | tr -d '\n')
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"baidu/Unlimited-OCR-2\",\"temperature\":0,\"max_tokens\":8192,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"document parsing.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}}]}]}"
```

百度通用文字识别（官方表单；`--data-urlencode` 会把 base64 里的 `+` `/` `=` 编进表单）：

```bash
IMG=$(base64 -i page.png | tr -d '\n')
curl -s http://127.0.0.1:6590/rest/2.0/ocr/v1/general_basic \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "image=$IMG"
```

同一张图的高精度路径，JSON 正文：

```bash
curl -s http://127.0.0.1:6590/rest/2.0/ocr/v1/accurate_basic \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMG\"}"
```

阿里云 RecognizeGeneral。`Data` 是字符串，里面才是 `content` 和 `prism_wordsInfo`：

```bash
curl -s http://127.0.0.1:6590/ocr/RecognizeGeneral \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"Url":"https://example.com/page.png"}'
```

原始图片字节，或 `Action` 查询参数：

```bash
curl -s "http://127.0.0.1:6590/ocr?Action=RecognizeGeneral" \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: image/png" \
  --data-binary @page.png
```

腾讯云 GeneralBasicOCR / GeneralAccurateOCR：

```bash
curl -s http://127.0.0.1:6590/ocr/GeneralBasicOCR \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"ImageBase64\":\"$IMG\"}"

curl -s http://127.0.0.1:6590/ocr/GeneralAccurateOCR \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "X-TC-Action: GeneralAccurateOCR" \
  -H "Content-Type: application/json" \
  -d '{"ImageUrl":"https://example.com/page.png"}'
```

multipart 与聊天接口一样，字段名用 `image` 或 `file`：

```bash
curl -s http://127.0.0.1:6590/rest/2.0/ocr/v1/general \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -F "image=@page.png"
```

换 gen-1 权重：`-H "X-Nova-Model: mlx-community/Unlimited-OCR-8bit"`。提示词仍是 `document parsing.`。

## Limits / notes

- 结构对齐，便于把 host 指过来。不是各厂商每个字段、签名算法、错误码表的逐位复刻。
- 专用 OCR（证件、银行卡、发票、车牌、手写专项）不在这一版。
- `Confidence` / `prob` 为 `0`，表示模型没有给置信度，不是识别失败。
- 远程 URL 的框留在模型的 0…1000 网格里。字节随请求上来时才换成像素。
- PDF 的 base64 会落成临时文件，识别结束后删除。URL 必须以 `.pdf` 结尾才会走已有的 PDF 渲染，这和聊天接口一致。
- 采样与聊天 OCR 优化器相同：Unlimited OCR 缺省 temperature 0、`max_tokens` 8192。

## Related

- [Unlimited OCR](unlimited-ocr.md)
- [API 兼容性](api-compatibility.md)
- [视觉 / VLM](vision.md)
