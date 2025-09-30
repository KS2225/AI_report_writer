# 🔬 AI Research Agent

**Ask a question. Get a research report.**
An AI-powered research assistant that transforms any query into a structured, reference-backed markdown report — all within a sleek dark-mode Gradio interface.

---

## ✨ Features

* 🧠 **Smart Query Planning** – Expands your question into 20+ diverse sub-queries.
* 🔍 **Deep Search** – Gathers insights across perspectives, trends, and subtopics.
* 📝 **Structured Reports** – Generates a long markdown report.
* 🔮 **Next Steps** – Suggests follow-up research directions.

---

## 🚀 Demo

<video width="600" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## ⚡️ Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/ks2225/AI_report_writer.git
cd ai-research-agent
```

### Add API Keys

Make sure to enter your Google and SerpAPI api keys in the placeholders


### 4. Run the app

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🖥️ Usage

1. Enter your research question in the input box.
2. The system:

   * Expands your query into multiple search strategies.
   * Calls Google AI for results.
   * Compiles everything into a polished research report.
3. Copy the markdown output for notes, blogs, or papers.

---

## 🛠️ Tech Stack

* [Gradio](https://www.gradio.app/) – interactive web UI
* [Google Gemini 2.0 Flash](https://ai.google.dev/) – text generation + search reasoning
* Python (async/await for smooth requests)

---


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
