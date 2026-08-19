# loader

Reads documents from local file sources into a uniform `loader.Document`
(ID, title, source, content, metadata) ready for `store.Upload` or the
`ingest` pipeline. Loaders are pure readers: they do not chunk, embed, or
store.

## Loaders

| Loader | Format | Dependencies |
|--------|--------|--------------|
| `TextLoader` | `.txt` | stdlib |
| `MarkdownLoader` | `.md` | stdlib (heading-aware document splitting) |
| `CSVLoader` | `.csv` | stdlib |
| `JSONLoader` | `.json` | stdlib |
| `HTMLLoader` | `.html` | `golang.org/x/net` |
| `PDFLoader` | `.pdf` | `ledongthuc/pdf` (pure Go) |
| `DocxLoader` | `.docx` | stdlib (zip + XML) |
| `DirectoryLoader` | walks a tree, dispatches by extension | — |

`ForExtension(ext)` returns the default loader for an extension
(`UnsupportedExtError` otherwise). `DirectoryLoader` takes an explicit
extension list, a recursive flag, and optional per-extension overrides:

```go
l, err := loader.NewDirectoryLoader([]string{".md", ".txt"}, true, nil)
docs, err := l.Load(ctx, "/path/to/corpus")
```

`ExtractHTML` is exported for direct HTML text extraction.
