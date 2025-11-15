from huggingface_hub import login

login("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")
print("Login exitoso")
# src/agents.py

from huggingface_hub import InferenceClient
from langchain_community.tools import DuckDuckGoSearchRun

# ---------- CONFIGURACIÓN ----------
TOPIC = "Impact of synthetic data in healthcare"  # puedes cambiar el tema

# Usamos un modelo soportado para conversación (chat), no text_generation:
# Zephyr funciona para conversacional; evitamos errores de proveedor.
CHAT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# ---------- HERRAMIENTAS ----------
search_tool = DuckDuckGoSearchRun()
writer_client = InferenceClient(model=CHAT_MODEL)

# ---------- AGENTES ----------
class ResearcherAgent:
    def run(self):
        print("[Researcher] Buscando información...")
        query = f"{TOPIC} site:arxiv.org OR site:researchgate.net OR site:medium.com"
        results = search_tool.run(query)
        # LangChain devuelve texto; tomamos líneas útiles
        snippets = [s.strip() for s in results.split("\n") if s.strip()]
        top_snippets = snippets[:6] if snippets else ["No sources found."]
        print(f"[Researcher] Encontrados {len(top_snippets)} fragmentos.")
        return top_snippets

class WriterAgent:
    def run(self, snippets):
        print("[Writer] Generando resumen...")
        sources_text = "\n".join(snippets)

        # Construimos mensajes de chat para que el modelo responda de forma estructurada
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research writer. Use only provided sources. "
                    "Write a ~500-word Markdown report with the exact headers: "
                    "Introduction, Key Findings, Ethical & Technical Challenges, Conclusion. "
                    "Avoid hallucinations; be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {TOPIC}\n\nSources:\n{sources_text}\n\n"
                    "Write the full report now."
                ),
            },
        ]

        # Usamos chat_completion (no text_generation) para evitar errores de proveedor.
        try:
            # Stream para construir la respuesta progresivamente
            final = ""
            for chunk in writer_client.chat_completion(messages=messages, max_tokens=800, temperature=0.3, stream=True):
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.get("content"):
                        final += delta["content"]
            if not final:
                # Fallback sin streaming
                resp = writer_client.chat_completion(messages=messages, max_tokens=800, temperature=0.3)
                final = resp.choices[0].message["content"]
        except Exception as e:
            raise RuntimeError(f"[Writer] Error al generar texto: {e}")

        print("[Writer] Resumen generado.")
        return final

class ReviewerAgent:
    def run(self, text):
        print("[Reviewer] Evaluando estructura...")
        feedback = []
        if len(text) < 300:
            feedback.append("El texto es muy corto, debe tener ~500 palabras.")
        for header in ["Introduction", "Key Findings", "Ethical & Technical Challenges", "Conclusion"]:
            if header not in text:
                feedback.append(f"Falta la sección '{header}'.")

        return {"feedback": feedback, "ok": len(feedback) == 0}

# ---------- FLUJO DE TRABAJO ----------
def run_workflow():
    researcher = ResearcherAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()

    snippets = researcher.run()
    draft = writer.run(snippets)
    review = reviewer.run(draft)

    if review["ok"]:
        print("[Reviewer] Todo correcto. Usando versión original.")
        return draft

    print("[Reviewer] Corrigiendo según feedback...")
    fix_messages = [
        {
            "role": "system",
            "content": (
                "You revise research reports to meet structure and clarity requirements. "
                "Preserve correctness and add missing headers if needed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Feedback: {review['feedback']}\n\nDraft:\n{draft}\n\n"
                "Produce a final ~500-word Markdown report with the required headers."
            ),
        },
    ]

    final = ""
    for chunk in writer_client.chat_completion(messages=fix_messages, max_tokens=800, temperature=0.2, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.get("content"):
                final += delta["content"]
    if not final:
        resp = writer_client.chat_completion(messages=fix_messages, max_tokens=800, temperature=0.2)
        final = resp.choices[0].message["content"]

    return final

def save_markdown(text, path="research_summary.md"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Output] Guardado en: {path}")

# ---------- EJECUCIÓN DIRECTA ----------
if __name__ == "__main__":
    report = run_workflow()
    save_markdown(report)
