import os
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- Core Configuration Settings ---
PARSED_DIR = "./processed_markdowns"  # Where your .md and _meta.json files live
CHUNKS_DIR = "./processed_chunks"     # Destination directory for structural chunks


class SEBIChunkingPipeline:
    def __init__(self, output_dir: str = CHUNKS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Define the structural markdown headers we want to split by
        self.headers_to_split_on = [
            ("#", "Header_1"),       # Typically Document Title / Subject
            ("##", "Header_2"),      # Typically Chapters / Main Sections
            ("###", "Header_3"),     # Typically Specific Clauses / Subsections
            ("####", "Header_4")     # Detailed sub-bullets
        ]
        
        # Initialize the Markdown Splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False      # Keep headers inside the text chunk to preserve semantic context
        )
        
        # 2. Define secondary splitter to handle oversized clauses safely
        # Chunk size of 1000-1500 characters balances embedding density and context retention
        self.secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        print("⚡ SEBI Legal Chunking Pipeline Layer Initialized.")

    def chunk_single_document(self, md_path: Path, meta_path: Path) -> list:
        """
        Loads an isolated markdown file, inherits its corporate legal metadata, 
        and slices it into contextually complete text chunks.
        """
        doc_id = md_path.stem
        
        # Load local markdown text
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            
        # Load associated global metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            global_metadata = json.load(f)
            
        # Execute Layer 1: Markdown Structural Header Splitting
        md_header_splits = self.markdown_splitter.split_text(markdown_content)
        
        final_processed_chunks = []
        chunk_sequence = 0
        
        # Execute Layer 2: Recursive Sub-splitting for token window protection
        for element in md_header_splits:
            # Sub-split the text if a structural section is too long
            sub_documents = self.secondary_splitter.split_documents([element])
            
            for sub_doc in sub_documents:
                chunk_sequence += 1
                
                # Consolidate inherited markdown headers into a clean structural string path
                # e.g., "Master Circular -> Chapter II -> Registration Requirements"
                header_trail = [
                    str(sub_doc.metadata.get(f"Header_{i}", "")).strip()
                    for i in range(1, 5)
                    if str(sub_doc.metadata.get(f"Header_{i}", "")).strip()
                ]
                context_path = " > ".join(header_trail) if header_trail else "General Content"
                
                # Core Metric Reinforcement: Context Completeness (0 to 1)
                # We blend global descriptors, header context, and the layout text into a unified chunk payload
                chunk_payload = {
                    "chunk_id": f"{doc_id}_chunk_{chunk_sequence:03d}",
                    "doc_id": doc_id,
                    "text_content": sub_doc.page_content,
                    "metadata": {
                        "circular_number": global_metadata.get("circular_number", "UNKNOWN"),
                        "document_type": global_metadata.get("document_type", "UNKNOWN"),
                        "issuance_date": global_metadata.get("issuance_date", "UNKNOWN"),
                        "category": global_metadata.get("category", "General Legal"), # e.g. "Research Analysts (RA)"
                        "title": global_metadata.get("title", doc_id),
                        "header_context_path": context_path,
                        "chunk_index": chunk_sequence
                    }
                }
                final_processed_chunks.append(chunk_payload)
                
        return final_processed_chunks

    def process_all_parsed_files(self, input_dir: str):
        """
        Scans the parser's output directory, processes all valid file pairs,
        and saves chunks into a consolidated JSON manifest ready for database injection.
        """
        md_files = list(Path(input_dir).glob("*.md"))
        if not md_files:
            print(f"❌ No markdown files found in: {input_dir}. Run your parsing layer first.")
            return
            
        print(f"📂 Found {len(md_files)} parsed documents. Commencing legal contextual chunking...")
        
        all_system_chunks = []
        
        for md_path in md_files:
            meta_path = md_path.with_name(f"{md_path.stem}_meta.json")
            
            if not meta_path.exists():
                print(f"⚠️ Missing metadata manifest for {md_path.name}. Skipping file.")
                continue
                
            doc_chunks = self.chunk_single_document(md_path, meta_path)
            all_system_chunks.extend(doc_chunks)
            print(f" ↳ Sliced '{md_path.stem}' into {len(doc_chunks)} context-complete chunks.")
            
        # Save independent files for inspection
        chunks_output_filepath = os.path.join(self.output_dir, "sebi_rag_chunks_manifest.json")
        with open(chunks_output_filepath, "w", encoding="utf-8") as f:
            json.dump(all_system_chunks, f, indent=4)
            
        print(f"\n✅ Chunking Complete! Compiled {len(all_system_chunks)} total chunks.")
        print(f"📦 Combined chunks manifest exported to: '{chunks_output_filepath}'")


# --- Pipeline Driver Execution ---
if __name__ == "__main__":
    # Point this to your parser's output cache directory
    pipeline = SEBIChunkingPipeline(output_dir=CHUNKS_DIR)
    pipeline.process_all_parsed_files(input_dir=PARSED_DIR)