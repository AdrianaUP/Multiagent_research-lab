# Multiagent_research-lab

Este proyecto implementa un flujo de trabajo colaborativo entre tres agentes:

- **ResearcherAgent**: busca información relevante usando DuckDuckGo.
- **WriterAgent**: genera un resumen estructurado usando Hugging Face.
- **ReviewerAgent**: evalúa la estructura y corrige si es necesario.

## Archivos principales

- `src/agents.py`: implementación de los agentes
- `research_summary.md`: resumen generado
- `notebooks/workflow_demo.ipynb`: demostración del flujo

## Requisitos

- Python 3.10+
- `huggingface_hub`, `langchain_community`, `ddgs`

## Ejecución

```bash
.venv\Scripts\activate
python src/agents.py
