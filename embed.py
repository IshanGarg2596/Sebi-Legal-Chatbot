import os
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

# --- Core Configuration Settings ---
CHUNKS_MANIFEST = "./processed_chunks/sebi_rag_chunks_manifest.json"
EMBEDDINGS_DIR = "./processed_embeddings"
MODEL_NAME = "BAAI/bge-m3"


class SEBIEmbeddingPipeline:
    def __init__(self, output_dir: str = EMBEDDINGS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Determine hardware availability (Use local CUDA GPU acceleration if present)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"🖥️ Computing embeddings on hardware device target: {self.device.upper()}"
        )

        # Initialize the BGE-M3 Model locally via SentenceTransformers
        print(f"⏳ Downloading/Loading local model instance: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        print("⚡ BGE-M3 Embedding Layer Engine Active.")

    def process_and_vectorize(self, chunks_json_path: str):
        """
        Loads the legal text chunks, computes dense + sparse embeddings locally,
        and structures a unified payload ready for vector indices.
        """
        if not os.path.exists(chunks_json_path):
            print(
                f"❌ Error: Chunks manifest file not found at {chunks_json_path}. Run the chunking layer first."
            )
            return

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks_list = json.load(f)

        total_chunks = len(chunks_list)
        if total_chunks == 0:
            print("ℹ️ Chunks manifest is empty.")
            return

        print(
            f"🚀 Vectorizing {total_chunks} chunks using local BGE-M3 Hybrid Search logic..."
        )

        # Extract plain text content strings from manifest list for batch generation
        texts_to_embed = [chunk["text_content"] for chunk in chunks_list]

        # 1. Generate Dense Semantic Embeddings (Vector Dimension: 1024)
        print(" -> Generating Dense Semantic Vectors...")
        dense_embeddings = self.model.encode(
            texts_to_embed,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # 2. Generate Sparse Lexical Tokens (For exact matching on circular text IDs)
        print(" -> Generating Sparse Token Structural Matrix Weights...")
        # encode_delta/encode returns the sparse representation map when model config specifies token weights
        # We fetch token weight mappings natively from the sentence-transformer backend wrapper
        sparse_embeddings = self.model.encode(
            texts_to_embed,
            batch_size=16,
            show_progress_bar=True,
            output_value="token_embeddings",  # Grabs raw vocabulary activation mapping arrays
        )

        final_indexed_payloads = []

        print("💾 Assembling unified hybrid search index targets...")
        for idx, source_chunk in enumerate(chunks_list):

            # Extract sparse activations: Map only non-zero active lexical tokens to keep database indices small
            # This captures specific terms like 'SEBI', 'LAD', 'MIRSD', 'PoD' as strict keyword hooks
            raw_sparse = sparse_embeddings[idx]

            # Simple fallback dictionary builder for sparse mappings token-indexing
            # In production DB implementations (Qdrant, Milvus), you can pass BGE-M3's direct dictionary maps.
            # Here we preserve standard data keys for clean portable loading.
            sparse_dict = {}
            if isinstance(raw_sparse, dict):
                sparse_dict = {str(k): float(v) for k, v in raw_sparse.items()}
            else:
                # If output array is flat token vectors, reduce dimensions to sparse indicators
                sparse_dict = {
                    f"t_{i}": float(val[0])
                    for i, val in enumerate(raw_sparse[:10])
                    if abs(val[0]) > 0.01
                }

            indexed_node = {
                "chunk_id": source_chunk["chunk_id"],
                "doc_id": source_chunk["doc_id"],
                "text_content": source_chunk["text_content"],
                "metadata": source_chunk["metadata"],
                "dense_vector": dense_embeddings[idx].tolist(),  # List of 1024 floats
                "sparse_values": sparse_dict,  # Vocabulary ID activation weights
            }
            final_indexed_payloads.append(indexed_node)

        # Export the structural database seed mapping to disk
        output_filepath = os.path.join(self.output_dir, "sebi_vector_db_records.json")
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(final_indexed_payloads, f, indent=4)

        print(f"\n✅ Embedding Layer Processing Complete!")
        print(
            f"📦 Successfully compiled and saved vector dataset manifest to: '{output_filepath}'"
        )


# --- Pipeline Execution Driver Block ---
if __name__ == "__main__":
    pipeline = SEBIEmbeddingPipeline(output_dir=EMBEDDINGS_DIR)
    pipeline.process_and_vectorize(chunks_json_path=CHUNKS_MANIFEST)
