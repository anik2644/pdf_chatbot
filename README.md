<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=800&color=00D4FF&background=F9FF0200&center=true&vCenter=true&width=700&height=65&lines=AI+Tour+Assistant+Chatbot;Context-Aware+RAG+System;Bangladesh+Tour+Spots)](https://git.io/typing-svg)

# 🧭 AI Tour Assistant — Bangladesh
### *Context-Aware RAG Chatbot for Travel Discovery*

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0078D4?style=for-the-badge&logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-TinyLlama-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> **An advanced AI-powered context-aware Q&A system** that answers Bangladesh tourism queries using semantic search over a structured PDF knowledge base — grounded in real data, not hallucinations.

<br/>

[🚀 Quick Start](#-installation--setup) · [🧩 How It Works](#-how-it-works) · [🗺️ Knowledge Base](#%EF%B8%8F-knowledge-base) · [📈 Roadmap](#-future-improvements)

---

</div>

## 🌟 What Makes This Different?

Traditional search systems fail at understanding **intent and context**. This system doesn't.

| Traditional Search | 🤖 This System |
|---|---|
| Keyword matching | Semantic understanding |
| Stateless queries | Conversational memory |
| Brittle exact matches | Handles vague & indirect questions |
| No source tracking | Full source attribution |
| Slow cold retrieval | Cached embeddings for speed |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

**🧠 Context-Aware Memory**
Understands follow-up queries by retaining conversational context across turns.

**🔍 Smart Query Understanding**
Handles vague, indirect, and intent-based questions — not just exact phrases.

**📄 PDF Knowledge Retrieval**
Extracts and processes structured + unstructured tourism data directly from PDFs.

</td>
<td width="50%">

**🎯 Semantic Search**
Uses sentence embeddings to retrieve *meaningfully relevant* content — not just keyword hits.

**📌 Anti-Hallucination Citations**
Every answer is grounded with an exact source reference (page + line).

**⚡ Caching & Optimization**
Efficient embedding storage and fast retrieval pipeline for real-time response.

</td>
</tr>
</table>

---

## 🧩 How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Embedding Model    │  ← sentence-transformers/all-MiniLM-L6-v2
│  (Query → Vector)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FAISS Vector DB   │  ← Semantic similarity search
│  (Top-K Retrieval)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Injection  │  ← Retrieved chunks + conversation history
│  into LLM Prompt    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   TinyLlama LLM     │  ← Generates grounded, human-like answer
│   Response + Source │
└─────────────────────┘
```

**Pipeline Steps:**

1. 📥 Load and preprocess PDF knowledge base
2. ✂️ Chunk text into semantically coherent segments
3. 🔍 Generate embeddings for each chunk
4. 📚 Index embeddings into FAISS vector database
5. 💬 User submits a natural language query
6. 🧠 Query is vectorized using the same embedding model
7. 🎯 Top-K relevant context chunks are retrieved
8. 🔗 Context + history injected into the LLM prompt
9. 🤖 Structured, cited answer is returned to the user

---

## 🗺️ Knowledge Base

> A curated **Bangladesh Tourism PDF dataset** covering 10 major destinations — structured for high-recall semantic retrieval.

<details>
<summary><b>🏝️ Cox's Bazar</b> — World's Longest Natural Sea Beach</summary>

Cox's Bazar stretches over **120 kilometers** along the Bay of Bengal — the longest unbroken natural sea beach on earth. Key highlights include **Himchari National Park** (waterfalls + mountainous terrain) and **Inani Beach** (coral and stone formations visible at low tide). A vivid blend of natural beauty and authentic fishing culture.

</details>

<details>
<summary><b>🌳 The Sundarbans</b> — UNESCO World Heritage Mangrove Forest</summary>

The **largest mangrove forest in the world**, the Sundarbans is home to the iconic **Royal Bengal Tiger**, 260+ bird species, crocodiles, and spotted deer. Tourism centers on boat cruises through tidal waterways and visits to wildlife sanctuaries like **Kotka** and **Kochikhali**.

</details>

<details>
<summary><b>🍃 Srimangal</b> — Tea Capital of Bangladesh</summary>

Nestled among rolling hills and lush tea estates, Srimangal is famed for its **Seven-Layer Tea** and the biodiversity of **Lawachara National Park** — home to the endangered **Western Hoolock Gibbon**. Also features the serene **Madhabpur Lake**.

</details>

<details>
<summary><b>🏞️ Rangamati</b> — Lake District of the Hill Tracts</summary>

Built around **Kaptai Lake** — Bangladesh's largest artificial lake — Rangamati offers the iconic **Hanging Bridge**, **Shuvalong Falls**, and immersive cultural experiences from **Chakma, Marma, and Tripura** indigenous communities.

</details>

<details>
<summary><b>⛰️ Bandarban</b> — Adventure Capital of Bangladesh</summary>

Bandarban is the go-to for trekkers and thrill-seekers, featuring **Nilgiri**, **Nilachal**, **Boga Lake**, and the stunning **Shoilo Propat** waterfall. Rich in indigenous culture and high-altitude scenic viewpoints.

</details>

<details>
<summary><b>🏝️ Saint Martin's Island</b> — Bangladesh's Only Coral Island</summary>

Locally known as **Narikel Jinjira**, this tiny island offers **snorkeling**, coral viewing, sea turtle nesting sites, and peaceful moonlit beaches — the most pristine marine destination in Bangladesh.

</details>

<details>
<summary><b>🌫️ Sajek Valley</b> — The Roof of Rangamati</summary>

Sajek is famous for its **cloud-filled valleys** and breathtaking sunrise panoramas. Indigenous communities — **Lushai, Pankho, and Tripura** — call this highland home. Best experienced during or just after the **monsoon season**.

</details>

<details>
<summary><b>🌅 Kuakata</b> — Daughter of the Sea</summary>

Unique in the world for offering **both sunrise and sunset views from the same beach**, Kuakata also features Buddhist temples, the **Fatra mangrove forest**, and traditional fishing villages along a vast shoreline.

</details>

<details>
<summary><b>🏛️ Paharpur</b> — UNESCO Ancient Buddhist Monastery</summary>

Home to **Somapura Mahavihara**, one of the largest ancient Buddhist monasteries in South Asia and a UNESCO World Heritage Site. Famous for intricate **Pala-era terracotta art** and remarkable historical architecture.

</details>

<details>
<summary><b>🏙️ Old Dhaka</b> — Mughal Heart of the Capital</summary>

A living museum of **Mughal and colonial history** — featuring **Ahsan Manzil** (the Pink Palace), **Lalbagh Fort**, **Star Mosque**, and **Sadarghat River Port**. The cultural and historical soul of Bangladesh's capital.

</details>

---

## 💬 Example Conversation

```
┌─────────────────────────────────────────────────────────────────┐
│  User  ›  Tell me about Cox's Bazar                             │
│                                                                  │
│  Bot   ›  Cox's Bazar is home to the world's longest natural    │
│           sea beach at 120km. Key attractions include Himchari  │
│           National Park and Inani Beach's coral formations.     │
│           📄 Source: Tourism Guide (Page 2, line 4)             │
├─────────────────────────────────────────────────────────────────┤
│  User  ›  What's the best time to visit there?                  │
│           (Follow-up — no location repeated)                    │
│                                                                  │
│  Bot   ›  The best time to visit Cox's Bazar is October–March,  │
│           when weather is pleasant for beach activities.        │
│           📄 Source: Tourism Guide (Page 2, line 13)            │
└─────────────────────────────────────────────────────────────────┘
```

> ✅ Notice: The second question never mentioned "Cox's Bazar" — the system resolved the reference from **conversational context**.

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **Embedding Model** | `all-MiniLM-L6-v2` | Semantic vector generation |
| **LLM** | `TinyLlama-1.1B-Chat` | Response generation |
| **Orchestration** | LangChain | RAG pipeline management |
| **Vector Store** | FAISS | Fast similarity search |
| **Document Loader** | PyPDF | PDF parsing & chunking |

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/anik2644/pdf_chatbot.git
cd pdf_chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the chatbot
python main.py
```

---

## 📈 Future Improvements

- [ ] 📚 **Multi-document support** — ingest multiple travel PDFs simultaneously
- [ ] 🎤 **Voice interaction** — speech-to-text input & TTS output
- [ ] 🌐 **Multilingual support** — English + বাংলা (Bangla)
- [ ] 👍 **User feedback loop** — thumbs up/down for retrieval quality
- [ ] 📊 **Evaluation dashboard** — accuracy & retrieval quality metrics
- [ ] 🗺️ **Interactive map integration** — visual destination explorer

---

## 🎯 Why This Project Stands Out

This isn't a simple chatbot wrapper. It's a full **Retrieval-Augmented Generation** pipeline that demonstrates:

- ✅ Context-aware multi-turn conversation
- ✅ Semantic search over a domain-specific knowledge base
- ✅ Grounded responses with anti-hallucination source attribution
- ✅ End-to-end NLP pipeline: embeddings → vector DB → LLM inference
- ✅ Real-world application in the Bangladesh tourism domain

---

<div align="center">

**Built with ❤️ by [anik2644](https://github.com/anik2644)**

*📝 MIT License — free to use, modify, and distribute*

</div>
