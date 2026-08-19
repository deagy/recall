# core

The foundation types every other package builds on: documents, chunks, the
`Value` metadata type, and sentinel errors. No dependencies outside the
standard library.

## Types

| Type | Purpose |
|------|---------|
| `Document` | A source document (ID, title, source, namespace, metadata). Create with `NewDocument(id, title, source)`. |
| `Chunk` | A chunked piece of a document with its embedding, index, and metadata. |
| `Value` | A typed JSON-like metadata value (`String`, `Number`, `Boolean`, `URI`, `Literal`). Convert with `ToString` / `ToBool` / `ToFloat64`. |
| `ErrNotFound`, `ErrInvalidDocument`, `ErrInvalidChunk` | Sentinel errors; match with `errors.Is`. |

## Minimal example

```go
doc := core.NewDocument("doc-1", "My Doc", "file.txt")
doc.Metadata["lang"] = core.String("en")

if v, ok := core.ToString(doc.Metadata["lang"]); ok {
    fmt.Println(v) // en
}
```

## Used by

`chunker`, `embedder`, `index`, `store`, `loader`, and the rest of the
pipeline all exchange `core.Document` / `core.Chunk` values.
