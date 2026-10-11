from pathlib import Path

agent_path = Path("backend/agent.py")
app_path = Path("backend/app.py")

agent = agent_path.read_text(encoding="utf-8")

start = agent.index("def llm_narrate(question: str, evidence: dict) -> str | None:")
end = agent.index("# ------------------------------------------------------------------------ agent", start)

replacement = '''def llm_narrate(question: str, evidence: dict) -> str | None:
    """Use Gemini to narrate grounded evidence; fall back to the template."""
    try:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()

        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key == "PASTE_YOUR_NEW_API_KEY_HERE":
            return None

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        ev = json.dumps(evidence, default=str)
        prompt = (
            "You are a renewable-energy operations analyst. Write a concise "
            "plain-English explanation in at most 120 words using only the "
            "JSON evidence below. Do not invent numerical claims. State "
            "confidence and the main caveat. Label possible causes as "
            "hypotheses, not facts.\\\\n\\\\n"
            f"Question: {question}\\\\n\\\\nEvidence JSON:\\\\n{ev}"
        )

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        return answer if answer and grounded(answer, ev) else None
    except Exception as exc:
        print(f"Gemini request failed: {type(exc).__name__}: {exc}")
        return None


'''

agent = agent[:start] + replacement + agent[end:]
agent = agent.replace(
    '* If ``ANTHROPIC_API_KEY`` is set, an LLM *rewrites* the evidence into a short analyst note.',
    '* If ``GEMINI_API_KEY`` is set, Gemini rewrites the evidence into a short analyst note.'
)
agent = agent.replace('DEFAULT_LLM_MODEL = "claude-sonnet-5-5"',
                      'DEFAULT_LLM_MODEL = "gemini-2.5-flash"')
agent_path.write_text(agent, encoding="utf-8")

app = app_path.read_text(encoding="utf-8")
if "from dotenv import load_dotenv" not in app:
    app = app.replace(
        "from pydantic import BaseModel, Field",
        "from pydantic import BaseModel, Field\nfrom dotenv import load_dotenv"
    )
if "\nload_dotenv()\n" not in app:
    app = app.replace(
        "BASE = Path(__file__).resolve().parents[1]",
        "load_dotenv()\n\nBASE = Path(__file__).resolve().parents[1]"
    )
app = app.replace(
    'bool(os.getenv("ANTHROPIC_API_KEY"))',
    'bool(os.getenv("GEMINI_API_KEY")) and os.getenv("GEMINI_API_KEY") != "PASTE_YOUR_NEW_API_KEY_HERE"'
)
app_path.write_text(app, encoding="utf-8")

print("Gemini integration updated. Backups were created.")
