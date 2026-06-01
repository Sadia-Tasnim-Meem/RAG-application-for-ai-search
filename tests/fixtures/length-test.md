# Text splitter strategies

## Text structure-based

Text is organized into paragraphs, sentences, and words.
The `RecursiveCharacterTextSplitter` implements this:
- Keeps larger units intact (paragraphs)
- Falls back to sentences, then words

## Length-based

An intuitive strategy: split documents by length.
Types of length-based splitting:
- Token-based: splits by number of tokens (useful for LLMs)
- Character-based: splits by number of characters (consistent across text)

## Document structure-based

Documents with structure are split by their format:
- Markdown: by headers
- HTML: by tags
- JSON: by objects or arrays
