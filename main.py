from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import json
from typing import List, Dict, Any
import os
from datetime import datetime

app = FastAPI(title="Scientific Searcher")

templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

model = SentenceTransformer('all-MiniLM-L6-v2')
papers_idx = faiss.read_index("data/papers.idx")
mapping = pd.read_parquet("data/id_mapping.parquet")

def search_papers(prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for papers related to the prompt."""
    # Encode the prompt
    emb = model.encode([prompt]).astype('float32')
    
    # Search in the index
    distances, indices = papers_idx.search(emb, top_k)
    
    # Get the top papers
    top_papers = mapping.iloc[indices[0]]
    target_ids = set(top_papers["id"])
    
    # Load the matching documents
    matching_docs = {}
    with open("data/arxiv.json", "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("id") in target_ids:
                    matching_docs[obj["id"]] = obj
            except json.JSONDecodeError:
                continue
    
    # Format results
    results = []
    for i, paper_id in enumerate(top_papers["id"]):
        if paper_id in matching_docs:
            paper = matching_docs[paper_id]
            
            # Parse authors
            authors = [a.strip() for a in paper.get("authors", "").split(",")]
            
            # Parse year from versions
            year = None
            if paper.get("versions"):
                try:
                    date_str = paper["versions"][0]["created"]
                    year = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT").year
                except:
                    pass
            
            results.append({
                "id": paper_id,
                "title": paper.get("title", "No title"),
                "abstract": paper.get("abstract", "No abstract available").replace("\n", " ").strip(),
                "authors": authors,
                "categories": paper.get("categories", "").split(),
                "year": year,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "similarity_score": float(1 - distances[0][i]),
                "update_date": paper.get("update_date", ""),
                "rank": i + 1
            })
    
    return results

def create_template_files():
    os.makedirs("templates", exist_ok=True)
    
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scientific Searcher</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .paper-card {
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .paper-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .loading {
            display: none;
        }
        .abstract {
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8 text-center">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🔬 Scientific Searcher</h1>
            <p class="text-gray-600">Discover relevant research papers in seconds</p>
        </header>
        
        <div class="max-w-3xl mx-auto mb-8 bg-white rounded-xl shadow-sm p-6">
            <form id="searchForm" action="/search" method="post" class="flex flex-col gap-3">
                <div class="flex flex-col md:flex-row gap-3">
                    <input type="text" name="prompt" id="prompt" placeholder="Search for papers about..." 
                           class="flex-grow px-4 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                           value="{{ prompt if prompt else '' }}" required>
                    <select name="top_k" id="top_k" 
                            class="px-4 py-3 w-32 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent">
                        {% for k in [5, 10, 15, 20, 25, 30] %}
                            <option value="{{ k }}" {% if top_k == k %}selected{% endif %}>{{ k }} results</option>
                        {% endfor %}
                    </select>
                </div>
                <button type="submit" 
                        class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-all flex items-center justify-center">
                    <span>Search</span>
                    <div id="loadingSpinner" class="loading ml-2">
                        <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </div>
                </button>
            </form>
        </div>

        {% if results %}
        <div class="results-container space-y-5">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">Top {{ top_k }} results for <span class="text-blue-600 font-medium">"{{ prompt }}"</span></h2>
            
            {% for paper in results %}
            <div class="paper-card bg-white rounded-xl shadow-sm overflow-hidden">
                <div class="p-6">
                    <div class="flex flex-col md:flex-row justify-between items-start gap-3 mb-3">
                        <h3 class="text-lg font-semibold text-gray-800 flex-1">{{ paper.title }}</h3>
                        <div class="flex items-center gap-2">
                            <span class="bg-blue-100 text-blue-800 text-sm font-medium px-3 py-1 rounded-full">
                                {{ "%.0f"|format(paper.similarity_score * 100) }}% Match
                            </span>
                            {% if paper.categories %}
                            <span class="bg-gray-100 text-gray-600 text-sm font-medium px-3 py-1 rounded-full">
                                {{ paper.categories[0] }}
                            </span>
                            {% endif %}
                        </div>
                    </div>
                    
                    <div class="flex flex-wrap gap-2 text-sm text-gray-600 mb-3">
                        {% if paper.authors %}
                        <div class="flex items-center">
                            <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                            </svg>
                            <span>{{ paper.authors|join(", ") }}</span>
                        </div>
                        {% endif %}
                        {% if paper.year %}
                        <div class="flex items-center">
                            <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                            </svg>
                            <span>{{ paper.year }}</span>
                        </div>
                        {% endif %}
                    </div>
                    
                    <p class="text-gray-600 mb-4 abstract">{{ paper.abstract }}</p>
                    
                    <div class="flex justify-between items-center text-sm">
                        <div class="text-gray-500 flex items-center">
                            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            {{ paper.id }}
                        </div>
                        <a href="{{ paper.url }}" target="_blank" 
                           class="bg-blue-50 text-blue-600 hover:bg-blue-100 px-4 py-2 rounded-md font-medium flex items-center transition-colors">
                            Read Paper
                            <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                            </svg>
                        </a>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
            {% if prompt %}
            <div class="text-center py-12 bg-white rounded-xl">
                <p class="text-gray-500">No results found for "{{ prompt }}"</p>
                <p class="text-sm text-gray-400 mt-2">Try different keywords or more specific terms</p>
            </div>
            {% endif %}
        {% endif %}
    </div>

    <footer class="mt-12 py-6 bg-white border-t border-gray-100 text-center">
        <p class="text-gray-600">&copy; 2024 Scientific Searcher - Accelerating Research Discovery</p>
    </footer>

    <script>
        const form = document.getElementById('searchForm');
        const spinner = document.getElementById('loadingSpinner');
        form.addEventListener('submit', () => {
            spinner.style.display = 'block';
        });
    </script>
</body>
</html>
    """
    
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request, 
    prompt: str = Form(...),
    top_k: int = Form(5)
):
    # Validate and constrain top_k between 1-50
    top_k = max(1, min(50, top_k))
    results = search_papers(prompt, top_k=top_k)
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "prompt": prompt, "top_k": top_k, "results": results}
    )

if __name__ == "__main__":
    create_template_files()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)