import argparse

from app.ingestion.loader import load_document
from app.ingestion.splitter import split_documents
from app.ingestion.store import create_vector_store, get_vector_store
from app.retrieval.chain import query_with_citation


def cmd_ingest(args):
    if args.source_type == "web":
        documents = load_document("web", urls=args.urls)
    else:
        documents = load_document("file", file_path=args.path)
    chunks = split_documents(documents)
    create_vector_store(chunks)
    print(f"Ingested {len(chunks)} chunks from {args.source_type} source.")


def cmd_query(args):
    vector_store = get_vector_store()
    if vector_store is None:
        print("No documents ingested yet. Run 'ingest' first.")
        return
    result = query_with_citation(args.question, vector_store)
    print(f"\nQuestion: {result.question}\n")
    print(f"Answer: {result.answer}\n")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            page_info = f" (page {c.page})" if c.page else ""
            print(f"  [{c.chunk_id}] {c.source}{page_info}")
            print(f"      {c.text[:100]}...")
    else:
        print("No citations.")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("source_type", choices=["file", "web"])
    ingest_parser.add_argument("--path", help="File path (for file source)")
    ingest_parser.add_argument("--urls", nargs="+", help="URLs (for web source)")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question", help="Question to ask")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
