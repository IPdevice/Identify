from ddgs import DDGS

def search(query):
    ddgs = DDGS()
    
    ddgs.threads = 20

    results = ddgs.text(
        query=query,
        region="us-en",
        safesearch="off",
        max_results=5,
        backend="auto"
    )

    return {
        "query": query,
        "summary": True,
        "count": len(results),
        "results": results
    }

if __name__ == "__main__":
    response = search("what's the iot device?")
    print(response)
