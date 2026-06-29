import os
import re
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from llama_cloud import LlamaCloud

from dotenv import load_dotenv

# Load the variables from the .env file into the system environment
load_dotenv()

# --- Core Configuration Settings ---
INPUT_DIR = "./PDFs"  # Directory where your raw PDFs live
OUTPUT_DIR = "./processed_markdowns"  # Target directory for parsed RAG outputs


class SEBIFolderPipelineParser:
    def __init__(self, api_key: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the official unified LlamaCloud client wrapper
        self.client = LlamaCloud(api_key=api_key)
        print("🚀 SEBI LlamaParse Pipeline Engine Initialized successfully.")

    def extract_legal_metadata(self, markdown_text: str) -> dict:
        """
        Parses metadata from the extracted Markdown output.
        Prioritizes case-sensitive SEBI circular formats, falling back to LAD regulations.
        """
        # Focus on the first ~4000 characters (typically covers the entire first page)
        first_page_chunk = markdown_text[:4000]

        # 1. Regex handles lowercase matching natively to catch variants like 'PoD'
        circ_match = re.search(
            r"(?:SEBI/|HO/)[a-zA-Z0-9_/: \t\(\)-]+", first_page_chunk
        )

        if circ_match:
            circular_number = circ_match.group(0)
            doc_type = "SEBI Circular"
        else:
            # 2. Fallback: Search for Legal Affairs Department Regulation codes (e.g., LAD-NRO, LAD-NGO)
            lad_match = re.search(r"LAD-[a-zA-Z0-9_/:\t\(\)-]+", first_page_chunk)
            if lad_match:
                circular_number = lad_match.group(0)
                doc_type = "SEBI Regulation (LAD)"
            else:
                circular_number = "UNKNOWN_REGULATORY_ID"
                doc_type = "General Legal"

        # 3. Extract Date formatting variants common in Indian legal updates
        date_match = re.search(
            r"([A-Za-z]+ \d{1,2}, \d{4}|\d{2}[./-]\d{2}[./-]\d{4})", first_page_chunk
        )
        issuance_date = date_match.group(0) if date_match else "UNKNOWN_DATE"

        # 4. Deduce legal category using rule-based classification heuristics
        category = "General Legal/SEBI Master"
        text_lower = first_page_chunk.lower()
        if "investment adviser" in text_lower or " ia " in text_lower:
            category = "Investment Advisers (IA)"
        elif "research analyst" in text_lower or " ra " in text_lower:
            category = "Research Analysts (RA)"

        return {
            "circular_number": circular_number,
            "document_type": doc_type,
            "issuance_date": issuance_date,
            "category": category,
        }

    async def parse_document_async(self, pdf_path: Path):
        """
        Ingests and orchestrates cloud OCR/parsing asynchronously.
        Checks for a local markdown cache file before hitting cloud servers.
        """
        filename = pdf_path.name
        doc_id = pdf_path.stem

        # Compute expected local destinations beforehand
        md_path = os.path.join(self.output_dir, f"{doc_id}.md")
        meta_path = os.path.join(self.output_dir, f"{doc_id}_meta.json")

        try:
            # --- CACHE CHECK CONDITION ---
            if os.path.exists(md_path):
                print(
                    f"💾 Found local cache for '{filename}'. Bypassing Cloud API call."
                )
                with open(md_path, "r", encoding="utf-8") as f:
                    full_markdown_text = f.read()
                source_engine = "Local-Cache-Re-parse"
            else:
                # --- NO CACHE FOUND: INITIALIZE LLAMAPARSE API TRANSACTION ---
                print(f"🔄 Ingesting '{filename}' into LlamaParse (Agentic Engine)...")

                # 1. Stream file payload into secure storage
                with open(pdf_path, "rb") as f:
                    uploaded_file = self.client.files.create(file=f, purpose="parse")

                # 2. Trigger asynchronous Agentic Parse job
                parse_result = self.client.parsing.parse(
                    file_id=uploaded_file.id,
                    tier="agentic",
                    version="latest",
                    expand=["markdown"],
                )

                # 3. Extract compiled markdown payload from completed run
                full_markdown_text = ""
                if parse_result.markdown and parse_result.markdown.pages:
                    full_markdown_text = "\n\n".join(
                        [
                            page.markdown
                            for page in parse_result.markdown.pages
                            if page.markdown
                        ]
                    )

                if not full_markdown_text.strip():
                    print(f"⚠️ Warning: LlamaParse yielded empty text for {filename}.")
                    return

                # Save structural Markdown file locally for caching next time
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(full_markdown_text)

                source_engine = "LlamaParse-Agentic-V2"

            # --- UNIFIED LOCAL PARSING & LEDGER DUMP ---
            extracted_meta = self.extract_legal_metadata(full_markdown_text)

            final_payload = {
                "doc_id": doc_id,
                "title": doc_id.replace("_", " ").title(),
                "circular_number": extracted_meta["circular_number"],
                "document_type": extracted_meta["document_type"],
                "issuance_date": extracted_meta["issuance_date"],
                "category": extracted_meta["category"],
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "source_engine": source_engine,
            }

            # Write tracking metadata json manifest files
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(final_payload, f, indent=4)

            print(
                f"✨ Processed: {doc_id} -> ID: {extracted_meta['circular_number']} [{source_engine}]"
            )

        except Exception as e:
            print(f"❌ Exception encountered while processing {filename}: {str(e)}")


# --- Production Entry Point For Running in a .py File ---
async def main():
    # Retrieve API Key from your system environment configuration
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print(
            "CRITICAL ERROR: Please export LLAMA_CLOUD_API_KEY before running this pipeline script."
        )
        return

    # Instantiate pipeline engine
    folder_parser = SEBIFolderPipelineParser(api_key=api_key, output_dir=OUTPUT_DIR)

    # Gather all target PDF files in the target path
    pdf_targets = list(Path(INPUT_DIR).glob("*.pdf"))

    if not pdf_targets:
        print(f"❌ No PDF documents found in target folder: {INPUT_DIR}")
    else:
        print(
            f"📂 Found {len(pdf_targets)} PDFs. Dispatching parallel parsing pipeline execution..."
        )

        # asyncio.gather launches processing on all files concurrently
        await asyncio.gather(
            *(folder_parser.parse_document_async(pdf) for pdf in pdf_targets)
        )

        print(
            f"\n✅ Pipeline Sync Complete! Check elements inside folder: '{OUTPUT_DIR}'"
        )


if __name__ == "__main__":
    # Bootstraps the root asynchronous loop safely inside a standard python environment execution
    asyncio.run(main())
