# 🎬 Manim AI Agent Generator

An intelligent, multi-agent application that transforms any educational topic, mathematical concept, or computer science algorithm into a professional, high-quality video animation using **LangGraph**, **Google Gemini**, and **Manim**. It also features automated voiceover generation via **manim-voiceover** and gTTS.

---

## 🌟 Key Features

* **🤖 LangGraph Multi-Agent Workflow**: Orchestrates a robust pipeline including storyboard extraction, code generation, layout optimization, static analysis, self-healing error correction, and video rendering.
* **🧠 Gemini-Powered Intelligence**: Leverages Google Gemini (`gemini-3.6-flash`) to design comprehensive storyboards and write accurate Python Manim code.
* **🎙️ Synchronized Voiceovers**: Integrates `manim-voiceover` with Google Text-to-Speech (gTTS) to automatically sync audio narration with visual animations (`tracker.duration`).
* **🔄 Self-Healing Validation Loop**: Automatically checks generated code via Python AST parsing and strict Manim rule validation, catching common errors (e.g., using `CENTER` instead of `ORIGIN`, or forbidden `self.clear()` calls) and attempting fixes up to 3 times.
* **🖥️ Interactive Streamlit UI**: Provides a user-friendly dashboard to configure complexity, target duration, render quality, and voiceover options.

---

## 🛠️ Tech Stack

* **Frontend / UI**: [Streamlit](https://streamlit.io/)
* **Agentic Workflow**: [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://www.langchain.com/)
* **AI Model**: Google GenAI (`ChatGoogleGenerativeAI`)
* **Animation Engine**: [Manim Community](https://www.manim.community/) & `manim-voiceover`
* **Data Validation**: [Pydantic](https://docs.pydantic.dev/)

---

## 📋 Prerequisites & System Dependencies

Because this application runs **Manim**, your system must have the required underlying dependencies installed:

1. **Python 3.10+**
2. **FFmpeg**: Required for video rendering.
3. **LaTeX** (Optional but recommended for rendering mathematical equations cleanly).

For detailed Manim installation instructions across platforms (macOS, Linux, Windows), consult the [Official Manim Installation Guide](https://docs.manim.community/en/stable/installation.html).

---

## 📦 Installation

1. **Clone or download** the repository containing `app.py`.

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate