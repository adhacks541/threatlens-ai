import argparse
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="ThreatLens API CLI Demo Query Tool")
    parser.add_argument("query", type=str, help="Threat intelligence query to search for")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--url", type=str, default="http://localhost:8000/query", help="API Endpoint URL")
    args = parser.parse_args()
    
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "stream": args.stream
    }
    
    print(f"[*] Sending query: '{args.query}' to {args.url}")
    
    if args.stream:
        try:
            with requests.post(args.url, json=payload, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
            print()
        except requests.exceptions.RequestException as e:
            print(f"[!] Connection Error: {e}")
    else:
        try:
            response = requests.post(args.url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            print("\n[+] Retrieved Documents:")
            for i, doc in enumerate(data.get("retrieved_documents", [])):
                print(f"  {i+1}. ID: {doc.get('id', 'N/A')} | Score: {doc.get('score', 0):.4f}")
                
            print("\n[+] LLM Answer:")
            print("-" * 50)
            print(data.get("answer", "No answer provided."))
            print("-" * 50)
            
            if data.get("precision_at_k_example") is not None:
                print(f"\n[i] Precision@{args.top_k}: {data['precision_at_k_example']}")
                
        except requests.exceptions.RequestException as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
