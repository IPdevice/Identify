#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import numpy as np
from openai import AsyncOpenAI
import httpx
import asyncio

# -------------------------------
# Configuration
# -------------------------------
from config import API_KEY_OPENAI as API_KEY, BASE_URL_OPENAI as BASE_URL
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4.1-nano"

# Initialize async clients
_http_client = None
_llm_client = None
_embedding_client = None

def _get_clients():
    """Get or initialize API clients"""
    global _http_client, _llm_client, _embedding_client
    
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10),
            timeout=httpx.Timeout(30.0)
        )
    
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=_http_client
        )
    
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=_http_client
        )
    
    return _llm_client, _embedding_client


async def _get_embedding(text: str) -> List[float]:
    """Get embedding vector for text"""
    llm_client, embedding_client = _get_clients()
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Warning] Failed to get embedding: {e}, using empty vector")
        return [0.0] * 1536  # ada-002 dimension


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def mmr_selection(
    query_embedding: List[float],
    candidate_items: List[Any],
    candidate_embeddings: List[List[float]],
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[int]:
    """
    MMR (Maximal Marginal Relevance) algorithm to select most relevant and diverse items
    
    Args:
        query_embedding: Embedding vector of the query
        candidate_items: List of candidate items
        candidate_embeddings: List of embedding vectors for candidate items
        lambda_param: Parameter to balance relevance and diversity (0-1, 0.5 = balanced)
        top_k: Number of top items to select
    
    Returns:
        List of indices of selected items
    """
    if len(candidate_items) == 0:
        return []
    
    if len(candidate_items) <= top_k:
        return list(range(len(candidate_items)))
    
    selected_indices = []
    remaining_indices = set(range(len(candidate_items)))
    
    # Step 1: Select the most relevant item to the query
    relevance_scores = [
        _cosine_similarity(query_embedding, emb)
        for emb in candidate_embeddings
    ]
    first_idx = max(range(len(relevance_scores)), key=lambda i: relevance_scores[i])
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Subsequent steps: Use MMR formula for selection
    # MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    while len(selected_indices) < top_k and remaining_indices:
        best_score = float('-inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Calculate relevance to query
            relevance = _cosine_similarity(query_embedding, candidate_embeddings[idx])
            
            # Calculate maximum similarity to selected items (for diversity measurement)
            max_similarity = 0.0
            for selected_idx in selected_indices:
                similarity = _cosine_similarity(
                    candidate_embeddings[idx],
                    candidate_embeddings[selected_idx]
                )
                max_similarity = max(max_similarity, similarity)
            
            # MMR score: balance relevance and diversity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    return selected_indices


async def _summarize_with_llm(content_text: str, is_query: bool = False) -> str:
    """Summarize content directly using LLM"""
    llm_client, _ = _get_clients()
    
    if is_query:
        prompt = f"Please summarize the following JSON data in a concise and clear way:\n{content_text}"
    else:
        prompt = f"Please provide a concise summary of the following information:\n{content_text}"
    
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error summarizing: {str(e)}]"


async def summarize_item(item: dict, max_iter: int = 3) -> str:
    """
    Summarize a single JSON object or entire JSON structure
    Note: max_iter parameter is retained for compatibility but not used in this simplified implementation
    """
    try:
        start_time = time.time()
        
        # Convert dict to JSON string
        json_text = json.dumps(item, indent=2, ensure_ascii=False)
        
        print(f"[MMR Implementation] Starting single item summarization, length: {len(json_text)}")
        
        # Directly summarize using LLM
        result = await _summarize_with_llm(json_text, is_query=True)
        
        elapsed = time.time() - start_time
        print(f"[MMR Implementation] Summarization completed, elapsed time: {elapsed:.2f} seconds")
        
        return result
    except Exception as e:
        import traceback
        print(f"[MMR Implementation] Error summarizing: {str(e)}")
        traceback.print_exc()
        return f"[Error summarizing item] {e}"


async def summarize_json_list_with_mmr(data_list: list, query_text: str = "", max_iter: int = 1, top_k: int = 5) -> str:
    """
    Select representative items from list using MMR algorithm and summarize
    """
    try:
        start_time = time.time()
        
        if len(data_list) == 0:
            return "No items to summarize."
        
        print(f"[MMR Implementation] Processing list with {len(data_list)} items")
        
        # If few items, summarize all directly
        if len(data_list) <= top_k:
            print(f"[MMR Implementation] Number of items <= {top_k}, summarizing all directly")
            selected_items = data_list
            selected_indices = list(range(len(data_list)))
        else:
            # Select representative items using MMR algorithm
            print(f"[MMR Implementation] Selecting {top_k} representative items using MMR algorithm")
            
            # Get query embedding (if query_text provided)
            if query_text:
                query_embedding = await _get_embedding(query_text)
            else:
                # If no query, use concatenated text of first 3 items as query
                query_text = "\n".join([json.dumps(item, ensure_ascii=False) for item in data_list[:3]])
                query_embedding = await _get_embedding(query_text)
            
            # Get embeddings for all candidate items
            print(f"[MMR Implementation] Calculating embeddings for {len(data_list)} items...")
            candidate_embeddings = []
            for i, item in enumerate(data_list):
                item_text = json.dumps(item, ensure_ascii=False)
                emb = await _get_embedding(item_text)
                candidate_embeddings.append(emb)
                if (i + 1) % 10 == 0:
                    print(f"[MMR Implementation] Processed {i + 1}/{len(data_list)} items")
            
            # MMR selection
            selected_indices = mmr_selection(
                query_embedding=query_embedding,
                candidate_items=data_list,
                candidate_embeddings=candidate_embeddings,
                lambda_param=0.5,  # Balance relevance and diversity
                top_k=top_k
            )
            selected_items = [data_list[i] for i in selected_indices]
            print(f"[MMR Implementation] MMR selected indices: {selected_indices}")
        
        # Summarize selected items
        selected_text = "\n\n====\n\n".join([
            json.dumps(item, indent=2, ensure_ascii=False)
            for item in selected_items
        ])
        
        # Output selected raw content
        print(f"\n[MMR Implementation] ===== Selected Raw Content ({len(selected_items)}/{len(data_list)} items) =====")
        for idx, (selected_idx, item) in enumerate(zip(selected_indices, selected_items)):
            print(f"\n--- Selected Item #{idx+1} (Original Index: {selected_idx}) ---")
            print(json.dumps(item, indent=2, ensure_ascii=False))
        print(f"\n[MMR Implementation] ===== End of Selected Content Output =====\n")
        
        summary = await _summarize_with_llm(selected_text, is_query=True)
        
        elapsed = time.time() - start_time
        print(f"[MMR Implementation] List summarization completed, elapsed time: {elapsed:.2f} seconds (selected {len(selected_items)}/{len(data_list)} items)")
        
        # Return summary and selected content information
        result_info = {
            "summary": summary,
            "selected_count": len(selected_items),
            "total_count": len(data_list),
            "selected_indices": selected_indices,
            "selected_items": selected_items
        }
        
        # Format result as string for compatibility with original interface
        formatted_result = f"""[MMR Summary]
Summary: {summary}

[Selection Information]
- Selected {len(selected_items)} items from {len(data_list)} total
- Selected indices: {selected_indices}
- JSON of selected items:
{selected_text}
"""
        
        return formatted_result
    except Exception as e:
        import traceback
        print(f"[MMR Implementation] Error summarizing list: {str(e)}")
        traceback.print_exc()
        # Fallback: summarize first few items directly
        try:
            fallback_items = data_list[:min(5, len(data_list))]
            fallback_text = "\n\n".join([json.dumps(item, ensure_ascii=False) for item in fallback_items])
            return await _summarize_with_llm(fallback_text, is_query=True)
        except:
            return f"[Error summarizing list] {e}"


# summarize_json_content function defined below (unified interface)


def summarize_json_list(data_list: list, max_iter: int = 1, max_workers: int = 5) -> str:
    """
    Synchronous version for compatibility with original interface
    """
    return asyncio.run(summarize_json_list_with_mmr(data_list, max_iter=max_iter))


async def summarize_json_data(file_path: str, max_iter: int = 1, max_workers: int = 5) -> str:
    """Read JSON from file and summarize (async version)"""
    try:
        async with open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        return await summarize_json_content(content, max_iter=max_iter, max_workers=max_workers)
    except Exception as e:
        return f"[Error opening file {file_path}] {e}"


def summarize_json_data_sync(file_path: str, max_iter: int = 1, max_workers: int = 5) -> str:
    """Read JSON from file and summarize (sync version for compatibility)"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return asyncio.run(summarize_json_content(content, max_iter=max_iter, max_workers=max_workers))
    except Exception as e:
        return f"[Error opening file {file_path}] {e}"


# -------------------------------
# Main Exported Function - Supports Async Calls
# -------------------------------
async def _summarize_json_content_impl(content, max_iter: int = 1, max_workers: int = 5):
    """Internal implementation for JSON content summarization"""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception as e:
            return f"[Error parsing JSON string] {e}"
    
    if isinstance(content, dict):
        return await summarize_item(content, max_iter=max_iter)
    elif isinstance(content, list):
        return await summarize_json_list_with_mmr(content, max_iter=max_iter)
    else:
        return "[Error] JSON content is neither dict nor list"


def summarize_json_content(content, max_iter: int = 1, max_workers: int = 5):
    """
    Summarize JSON content - returns coroutine in async environment (needs await)
    Automatically runs synchronously in sync environment
    """
    try:
        # Check if running in async environment
        loop = asyncio.get_running_loop()
        # Return coroutine if in async context (requires await)
        return _summarize_json_content_impl(content, max_iter, max_workers)
    except RuntimeError:
        # Run synchronously if not in async context
        return asyncio.run(_summarize_json_content_impl(content, max_iter, max_workers))


# -------------------------------
# Test Code
# -------------------------------
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test single dict
        test_dict = {"name": "test", "value": 123}
        result1 = await summarize_json_content(test_dict)
        print(f"Dict Summary: {result1}\n")
        
        # Test list
        test_list = [
            {"id": 1, "name": "item1", "desc": "description 1"},
            {"id": 2, "name": "item2", "desc": "description 2"},
            {"id": 3, "name": "item3", "desc": "description 3"},
        ]
        result2 = await summarize_json_content(test_list)
        print(f"List Summary: {result2}\n")
    
    asyncio.run(test())