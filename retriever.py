import os
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore


def _tokenize(text: str) -> list[str]:
    """Chinese-aware character bigram tokenizer for BM25.
    Splits Chinese characters into bigrams and keeps alphanumeric tokens."""
    clean = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    tokens = []
    chars = []
    for ch in clean:
        if ch == ' ':
            if chars:
                tokens.extend(_char_bigrams(chars))
                chars = []
        elif '\u4e00' <= ch <= '\u9fff':
            chars.append(ch)
        else:
            if chars:
                tokens.extend(_char_bigrams(chars))
                chars = []
            tokens.append(ch.lower())
    if chars:
        tokens.extend(_char_bigrams(chars))
    return tokens


def _char_bigrams(chars: list[str]) -> list[str]:
    if len(chars) == 1:
        return chars
    return chars + [chars[i] + chars[i+1] for i in range(len(chars) - 1)]


class Retriever:
    def __init__(self):
        print("正在加载检索器...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_store = VectorStore()
        print("检索器加载完成！")

    def retrieve(self, query, k=5):
        query_vec = self.model.encode([query])[0]
        results = self.vector_store.search(query_vec, k=k)
        return results


class HybridRetriever(Retriever):
    def __init__(self, enable_bm25=True):
        super().__init__()
        self.enable_bm25 = enable_bm25
        self._bm25 = None
        self._bm25_texts = []
        self._bm25_metadata = None
        if enable_bm25:
            self._build_bm25()

    def _build_bm25(self):
        print("构建 BM25 索引...")
        metadata = self.vector_store.metadata
        self._bm25_texts = []
        for _, row in metadata.iterrows():
            combined = f"{row['question']} {row['answer_chunk']}"
            self._bm25_texts.append(combined)
        tokenized = [_tokenize(t) for t in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_metadata = metadata
        print(f"BM25 索引完成！共 {len(tokenized)} 篇文档")

    def retrieve_bm25(self, query, k=10):
        tokenized = _tokenize(query)
        scores = self._bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:k]

        max_score = max(scores) if len(scores) > 0 else 1.0
        results = []
        for idx in top_indices:
            meta = self._bm25_metadata.iloc[idx]
            results.append({
                'score': float(scores[idx] / max_score) if max_score > 0 else 0.0,
                'chunk_id': meta['chunk_id'],
                'question': meta['question'],
                'answer_chunk': meta['answer_chunk'],
                'category': meta['category'],
            })
        return results

    def retrieve_hybrid(self, query, k=5, bm25_k=10, vec_k=10):
        vec_results = self.retrieve(query, k=vec_k)
        bm25_results = self.retrieve_bm25(query, k=bm25_k)

        rrf_scores = {}
        for rank, item in enumerate(vec_results):
            cid = item['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rank + 60)

        for rank, item in enumerate(bm25_results):
            cid = item['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rank + 60)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        id_to_item = {}
        for item in vec_results + bm25_results:
            cid = item['chunk_id']
            if cid not in id_to_item:
                id_to_item[cid] = item

        merged = []
        for cid, rrf in sorted_ids[:k]:
            item = dict(id_to_item[cid])
            item['score'] = round(rrf, 6)
            merged.append(item)
        return merged

    def retrieve_by_method(self, query, k=5, method='hybrid'):
        if method == 'vector':
            return self.retrieve(query, k=k)
        elif method == 'bm25':
            return self.retrieve_bm25(query, k=k)
        elif method == 'hybrid':
            return self.retrieve_hybrid(query, k=k)
        else:
            return self.retrieve(query, k=k)
