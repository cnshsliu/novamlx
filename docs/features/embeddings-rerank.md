# Embeddings & Reranking

**Pitch:** Embed text for retrieval, then rerank candidate passages with a cross-encoder.

文本嵌入与交叉编码器重排，入口分别是 `POST /v1/embeddings` 与 `POST /v1/rerank`。

## Catalog

| 功能 | 详情 |
|------|------|
| **EmbeddingContainer** | 可热加载的嵌入模型 |
| **架构** | BERT、XLM-Roberta、ModernBert、Qwen3-ForTextEmbedding、Siglip、NomicBert |
| **池化** | mean / cls / last-token |
| **端点** | `POST /v1/embeddings` |
| **RerankerContainer** | 交叉编码器 reranker |
| **端点** | `POST /v1/rerank`（候选 → 打分后的候选） |

多模态总表中的同一组架构见 [多模态](multi-modal.md)。

正文来自 `features.zh-CN.md`「嵌入与重排」；标题与 pitch 为英文。
