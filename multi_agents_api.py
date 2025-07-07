import os
import requests
import openai
from langchain.tools import tool
from bs4 import BeautifulSoup

@tool
def github_top_repos(language: str = "python") -> str:
    """
    Uses GitHub API to fetch top-starred repositories in a given language.
    """
    token = os.getenv("GITHUB_API_KEY")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"GitHub API error: {response.status_code}"
    items = response.json()["items"][:3]
    return "\n\n".join([f"🔗 {r['full_name']} — {r['html_url']}\n⭐ {r['stargazers_count']} stars" for r in items])


@tool
def stackoverflow_search(query: str) -> str:
    """
    Searches Stack Overflow for programming-related questions.
    Input: query (e.g., 'python recursion')
    Returns: Top 3 matching question titles with links. and engry emoji.
    """
    url = f"https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Stack Overflow API error: {response.status_code}"
    
    results = response.json().get("items", [])[:3]
    if not results:
        return "No Stack Overflow results found."

    return "\n\n".join([
        f"📌 {item['title']}\n🔗 {item['link']}" for item in results
    ])



@tool
def search_protein_info(keyword: str) -> str:
    """
    Searches UniProt (EBI) for protein-related information.
    Input: keyword (e.g., 'insulin')
    Returns: top 3 protein names and their summary.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": keyword,
        "format": "json",
        "size": 3
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return "❌ UniProt API error"

    results = response.json().get("results", [])
    if not results:
        return "😕 No protein results found"

    output = []
    for protein in results:
        name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
        organism = protein.get("organism", {}).get("scientificName", "Unknown")
        accession = protein.get("primaryAccession", "N/A")
        output.append(f"🧬 **{name}**\n🔹 Organism: {organism}\n🔗 [View UniProt](https://www.uniprot.org/uniprotkb/{accession})")

    return "\n\n".join(output)


@tool
def get_gene_info(gene: str) -> str:
    """
    Uses Ensembl API to fetch basic gene info (Homo sapiens).
    Input: gene symbol (e.g., 'BRCA1')
    """
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?content-type=application/json"
    response = requests.get(url)
    if response.status_code != 200:
        return f"❌ Ensembl API error for gene: {gene}"

    data = response.json()
    name = data.get("display_name", "Unknown")
    desc = data.get("description", "No description")
    location = f"{data.get('seq_region_name')}:{data.get('start')}-{data.get('end')}"

    return f"""🧬 **{name}**
📍 Location: {location}
📖 Description: {desc}"""



@tool
def get_global_covid_stats() -> str:
    """
    Fetches real-time global COVID-19 statistics from disease.sh.
    """
    url = "https://disease.sh/v3/covid-19/all"
    response = requests.get(url)

    if response.status_code != 200:
        return "❌ Error fetching data from Open Disease API."

    data = response.json()
    return f"""🌍 **Global COVID-19 Stats**\n\n"
            "🦠 Cases: {data['cases']:,}\n"
            "💀 Deaths: {data['deaths']:,}\n"
            "💉 Recovered: {data['recovered']:,}\n"
            "📈 Today’s Cases: {data['todayCases']:,}\n"
            "🧪 Tests Done: {data['tests']:,}\n"
        """


@tool
def openfda_drug_info(drug_name: str) -> str:
    """
    Fetches drug label information (brand, purpose, warnings) using OpenFDA API for a given drug name.
    """
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": f"active_ingredient:{drug_name}", "limit": 3}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"OpenFDA API error: {response.status_code}"
    data = response.json()
    results = data.get("results", [])
    if not results:
        return "No results found for the drug."

    output = []
    for item in results:
        brand = item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
        purpose = item.get("purpose", ["Not specified"])[0]
        warnings = item.get("warnings", ["No warnings"])[0]
        output.append(f"💊 **{brand}**\nPurpose: {purpose}\nWarnings: {warnings}")
    return "\n\n".join(output)



@tool
def wolframalpha_query(query: str) -> str:
    """
    Uses the WolframAlpha API to answer advanced math, physics, and factual queries.
    Example: 'integrate x^2', 'speed of light', 'solve x^2 - 4x + 4 = 0'
    """
    app_id = os.getenv("WOLFRAMALPHA_APP_ID")
    url = "https://api.wolframalpha.com/v1/result"
    params = {
        "i": query,
        "appid": app_id
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return f"🧠 WolframAlpha says: {response.text}"
    else:
        return f"❌ WolframAlpha API error: {response.status_code} — {response.text}"



HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
def query_huggingface_model(payload, model="facebook/bart-large-cnn"):
    """
    Query a Hugging Face model like summarization, sentiment, etc.
    Example models:
    - facebook/bart-large-cnn (summarization)
    - distilbert-base-uncased-finetuned-sst-2-english (sentiment)
    """
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        return f"API Error: {response.status_code} - {response.text}"
    
    return response.json()

@tool
def summarize_text(text: str) -> str:
    """
    Uses Hugging Face BART model to summarize text input.
    """
    result = query_huggingface_model({"inputs": text}, model="facebook/bart-large-cnn")
    return result[0]['summary_text']



openai.api_key = os.getenv("OPENAI_API_KEY")
def ask_openai(prompt, model="gpt-4"):
    """
    Sends a prompt to OpenAI's GPT model and returns the response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

@tool
def explain_ai_topic(query: str) -> str:
    """
    Uses OpenAI GPT to explain any AI/ML/Data Science topic in simple terms.
    """
    return ask_openai(f"Explain this in a beginner-friendly way:\n{query}")


