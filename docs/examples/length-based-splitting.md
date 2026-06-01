## Query: "what are the types of length based splitting?"

**Doc:** `data/test_markdown.md` original source: https://docs.langchain.com/oss/python/integrations/splitters 
**Section:** `## Length-based`

---

### Before (dense-only — Goal 1)

**Retrieved chunks (top-4):**
| # | Section | Relevant? |
|---|---|---|
| [1] | Length-based | Yes — has both Token + Character |
| [2] | Text structure-based | No — code example from wrong section |
| [3] | Text structure-based | No — paragraph from wrong section |
| [4] | General intro | No — tip section |

**Answer:** (varies between runs — model format issue)
> *"The type of length-based splitting mentioned in the given documents is **Token-based**. This strategy involves splitting text based on the number of tokens..."*
>
> Only "Token-based" mentioned. Missed "Character-based" even though chunk [1] contains both.

**Root cause:** Dense semantic search matched "splitting" + "length" across sections. 3 of 4 chunks are noise from unrelated sections, drowning out the correct section.

**Full response:** `length-based-splitting.json`

**Note:** The generation is also unreliable — the Qwen 0.5B model sometimes echoes the prompt instead of answering. This is a prompt formatting issue separate from retrieval and will be addressed alongside Goal 2.

---

### After (hybrid BM25 + vector + chat template fix)

See full decision log at `docs/decision-log.md`.

**Retrieved chunks (top-4 after reranker):**
| # | Section | Relevant? |
|---|---|---|
| [1] | Length-based | Yes — both Token + Character |
| [2] | Text structure-based | No |
| [3] | General intro (has "length" keyword) | Partial |
| [4] | Text structure-based | No |

**Answer** (correct):
> *"The types of length-based splitting include: 1. Token-based: Splits text based on the number of tokens... 2. Character-based: Splits text based on the number of characters..."*

**What changed:**
1. **Retrieval fix**: Hybrid BM25 + vector boosted the correct chunk via keyword match on "length", "splitting", "token", "character". Cross-encoder confirmed relevance.
2. **Generation fix**: `ChatPromptTemplate` + `ChatHuggingFace` wrapper applies Qwen's required chat template so the model understands it's in a Q&A conversation, not a completion task.
