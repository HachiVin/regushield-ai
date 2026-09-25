import os
import sys
import json
import time
import re
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.rag_chain import FinancialRAGChain

load_dotenv()

judge_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

FAITHFULNESS_PROMPT = PromptTemplate.from_template("""
Anda adalah AI Quality Auditor independen.
Konteks: {context}
Jawaban Sistem: {answer}
Beri penilaian angka 0.0 sampai 1.0 apakah Jawaban didukung oleh Konteks.
Output HANYA format JSON valid: {{"score": <angka>, "reason": "<alasan>"}}
""")

CONTEXT_RELEVANCE_PROMPT = PromptTemplate.from_template("""
Anda adalah AI Quality Auditor independen.
Pertanyaan: {question}
Konteks: {context}
Beri penilaian angka 0.0 sampai 1.0 apakah Konteks relevan dengan Pertanyaan.
Output HANYA format JSON valid: {{"score": <angka>, "reason": "<alasan>"}}
""")

def parse_judge_output(raw_output: str) -> Dict[str, Any]:
    try:
        match = re.search(r'\{.*\}', raw_output.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"score": 0.0, "reason": "Gagal parsing JSON"}
    except Exception:
        return {"score": 0.0, "reason": "Error"}

def run_rag_evaluation():
    print("=" * 60)
    print("MEMULAI EVALUASI SISTEM RAG (AUDIT METRICS BENCHMARK)")
    print("=" * 60)

    with open("eval/test_dataset.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag_chain = FinancialRAGChain()
    
    total_relevance = 0.0
    total_faithfulness = 0.0
    total_latency = 0.0

    for idx, tc in enumerate(test_cases, 1):
        q = tc["question"]
        print(f"\n[{idx}/{len(test_cases)}] Menguji: '{q}' ...")
        
        start_time = time.time()
        response_data = rag_chain.answer_query(q, filter_source="Semua Dokumen")
        total_latency += (time.time() - start_time)
        
        answer = response_data["answer"]
        sources = response_data["sources"]
        context_snippets = "\n".join([f"[{s['source']}]: {s['snippet']}" for s in sources])
        
        # Evaluasi dengan LLM Judge
        faith_res = judge_llm.invoke(FAITHFULNESS_PROMPT.format(context=context_snippets, answer=answer))
        faith_data = parse_judge_output(faith_res.content)
        
        rel_res = judge_llm.invoke(CONTEXT_RELEVANCE_PROMPT.format(question=q, context=context_snippets))
        rel_data = parse_judge_output(rel_res.content)
        
        total_relevance += float(rel_data.get('score', 0.0))
        total_faithfulness += float(faith_data.get('score', 0.0))
        
        time.sleep(4) # Menjaga agar tidak terkena limit API Groq

    # Kalkulasi Rata-rata
    avg_rel = (total_relevance / len(test_cases)) * 100
    avg_faith = (total_faithfulness / len(test_cases)) * 100
    avg_lat = total_latency / len(test_cases)

    print("\n" + "=" * 60)
    print("HASIL RANGKUMAN BENCHMARK EVALUASI")
    print("=" * 60)
    print(f"Rata-rata Latensi  : {avg_lat:.2f} detik")
    print(f"Context Relevance  : {avg_rel:.1f}%")
    print(f"Faithfulness Score : {avg_faith:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_rag_evaluation()