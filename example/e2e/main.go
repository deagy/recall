// Command e2e is an end-to-end tutorial for the recall library, covering the
// full lifecycle of a RAG application:
//
//  1. ingest a local Markdown corpus (directory loader -> ingest pipeline ->
//     SQLite store with dedup and progress reporting),
//  2. search it (vector similarity and hybrid BM25+vector),
//  3. answer a question with the RAG pipeline and a mock LLM backend,
//  4. evaluate retrieval quality against a ground-truth dataset
//     (Precision/Recall/MRR/NDCG@K), and
//  5. extract a knowledge graph and run multi-hop reasoning over it.
//
// The example is fully deterministic and offline: it uses
// embedder.NewMockEmbedder and llm.NewMockBackend, so it needs no API keys,
// network access, or configuration.
//
// Run it with:
//
//	go run ./example/e2e
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/eval"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/ingest"
	"github.com/deagy/recall/llm"
	"github.com/deagy/recall/loader"
	"github.com/deagy/recall/pipeline"
	"github.com/deagy/recall/reasoning"
	"github.com/deagy/recall/store"
)

// corpus documents, each with one distinctive fact the eval queries target.
var corpus = map[string]string{
	"go.md":     "Go is a statically typed, compiled programming language designed at Google in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson. It emphasizes simplicity, fast compilation, and built-in concurrency through goroutines and channels.",
	"python.md": "Python is a high-level, general-purpose programming language created by Guido van Rossum and first released in 1991. Its design philosophy emphasizes code readability through significant indentation.",
	"rust.md":   "Rust is a systems programming language focused on safety, speed, and concurrency. It guarantees memory safety without a garbage collector, making it popular for operating systems and embedded development.",
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	workDir, err := os.MkdirTemp("", "recall-e2e-*")
	if err != nil {
		log.Fatalf("creating work dir: %v", err)
	}
	defer os.RemoveAll(workDir)

	st, err := store.NewSQLiteStore(store.Config{
		Namespace: "docs",
		Embedder:  embedder.NewMockEmbedder(384),
	}, filepath.Join(workDir, "recall.db"))
	if err != nil {
		log.Fatalf("creating store: %v", err)
	}
	defer st.Close()

	ingestCorpus(ctx, st, workDir)
	search(ctx, st)
	answerQuestion(ctx, st)
	evaluate(ctx, st)
	explainGraph()

	fmt.Println("\ne2e tutorial complete")
}

// ingestCorpus writes the corpus to disk and runs the ingest pipeline over
// it: load -> dedup -> chunk -> embed -> index, with progress reporting.
func ingestCorpus(ctx context.Context, st store.Store, workDir string) {
	fmt.Println("=== 1. Ingest (directory loader -> ingest pipeline -> SQLite) ===")

	corpusDir := filepath.Join(workDir, "corpus")
	if err := os.MkdirAll(corpusDir, 0o755); err != nil {
		log.Fatalf("creating corpus dir: %v", err)
	}
	for name, content := range corpus {
		if err := os.WriteFile(filepath.Join(corpusDir, name), []byte(content), 0o644); err != nil {
			log.Fatalf("writing %s: %v", name, err)
		}
	}

	dirLoader, err := loader.NewDirectoryLoader([]string{".md"}, true, nil)
	if err != nil {
		log.Fatalf("creating directory loader: %v", err)
	}
	prog := ingest.NewProgress()
	prog.OnDocument = func(id, outcome string) {
		fmt.Printf("  doc %s: %s\n", id, outcome)
	}
	p, err := ingest.NewPipeline(ingest.Options{
		Store:    st,
		Loader:   dirLoader,
		Source:   corpusDir,
		Dedup:    ingest.NewDeduplicator(),
		Progress: prog,
	})
	if err != nil {
		log.Fatalf("creating ingest pipeline: %v", err)
	}
	res, err := p.Run(ctx)
	if err != nil {
		log.Fatalf("running ingest: %v", err)
	}
	fmt.Printf("loaded=%d uploaded=%d skipped=%d failed=%d in %s\n",
		res.Loaded, res.Uploaded, res.Skipped, len(res.Failed), res.Duration.Round(time.Millisecond))
	fmt.Printf("store now holds %d chunks\n\n", st.Count())
}

// search demonstrates plain vector search and hybrid (BM25+vector) search.
func search(ctx context.Context, st store.Store) {
	fmt.Println("=== 2. Search (vector + hybrid) ===")

	modes := []struct {
		name string
		run  func(ctx context.Context, q string, opts index.SearchOptions) ([]index.SearchResult, error)
	}{
		{"vector", st.Search},
		{"hybrid", st.SearchHybrid},
	}
	for _, mode := range modes {
		results, err := mode.run(ctx, "memory safe programming language", index.SearchOptions{TopK: 2})
		if err != nil {
			log.Fatalf("%s search: %v", mode.name, err)
		}
		fmt.Printf("%s results for 'memory safe programming language':\n", mode.name)
		for i, r := range results {
			fmt.Printf("  %d. score=%.4f %s\n", i+1, r.Score, truncate(r.Chunk.Content, 60))
		}
	}
	fmt.Println()
}

// answerQuestion runs the RAG pipeline (retrieve -> assemble context ->
// render prompt) and feeds the rendered prompt to a mock LLM backend.
func answerQuestion(ctx context.Context, st store.Store) {
	fmt.Println("=== 3. RAG (pipeline + mock LLM) ===")

	rag := pipeline.NewRAGPipeline(st, nil).
		WithTopK(3).
		WithCitations()
	resp, err := rag.Query(ctx, "Which company designed the Go programming language?")
	if err != nil {
		log.Fatalf("rag query: %v", err)
	}
	fmt.Printf("retrieved %d source chunks, context ~%d tokens\n", len(resp.Sources), resp.Tokens)
	for _, c := range resp.Citations {
		chunk, ok := st.GetChunk(c.ChunkID)
		if !ok {
			continue
		}
		fmt.Printf("  citation %d: %s (%s)\n", c.Number, c.DocumentRef, truncate(chunk.Content, 50))
	}

	// In production this would be a real llm.Backend (OpenAI, Ollama, ...).
	backend := llm.NewMockBackend()
	backend.Response = "Go was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson."
	chat, err := backend.Chat(ctx, &llm.ChatRequest{
		Model:    "mock",
		Messages: []llm.Message{{Role: "user", Content: resp.Answer}},
	})
	if err != nil {
		log.Fatalf("llm chat: %v", err)
	}
	fmt.Printf("answer: %s\n\n", chat.Message.Content)
}

// storeRetriever adapts store.Store to eval.Retriever.
type storeRetriever struct{ s store.Store }

func (r storeRetriever) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	results, err := r.s.SearchHybrid(ctx, query, index.SearchOptions{TopK: k})
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(results))
	for _, res := range results {
		ids = append(ids, res.Chunk.ID)
	}
	return ids, nil
}

// evaluate builds a small ground-truth dataset and scores the store's
// hybrid retrieval against it.
func evaluate(ctx context.Context, st store.Store) {
	fmt.Println("=== 4. Evaluation (precision / recall / MRR / NDCG@K) ===")

	// Anchor each query to its ground-truth chunk with a distinctive phrase
	// from the target document.
	anchor := func(phrase string) string {
		results, err := st.SearchHybrid(ctx, phrase, index.SearchOptions{TopK: 1})
		if err != nil || len(results) == 0 {
			log.Fatalf("anchoring %q: %v", phrase, err)
		}
		return results[0].Chunk.ID
	}

	ds := eval.NewDataset("e2e-tutorial")
	ds.Add(eval.EvalQuery{
		Query:       "Which company designed the Go programming language?",
		RelevantIDs: []string{anchor("designed at Google in 2007")},
	})
	ds.Add(eval.EvalQuery{
		Query:       "Who created Python and when was it first released?",
		RelevantIDs: []string{anchor("created by Guido van Rossum")},
	})
	ds.Add(eval.EvalQuery{
		Query:       "What programming language guarantees memory safety without a garbage collector?",
		RelevantIDs: []string{anchor("memory safety without a garbage collector")},
	})

	suite := eval.NewBenchmarkSuite(ds, 5)
	report, err := suite.Run(ctx, storeRetriever{s: st})
	if err != nil {
		log.Fatalf("running evaluation: %v", err)
	}
	fmt.Printf("dataset=%s k=%d queries=%d\n", report.Dataset, report.K, report.NumQueries)
	fmt.Printf("precision@k=%.3f recall@k=%.3f mrr=%.3f ndcg@k=%.3f\n\n",
		report.MeanPrecision, report.MeanRecall, report.MeanMRR, report.MeanNDCG)
}

// explainGraph extracts a knowledge graph from text, finds a path between
// two entities, and runs the multi-hop reasoning engine over the graph.
func explainGraph() {
	fmt.Println("=== 5. Knowledge graph + multi-hop reasoning ===")

	text := "Alice works at Google. Bob studied at Stanford. Alice knows Bob."

	gs := store.NewMemoryGraphStore()
	bg := context.Background()
	entities, err := gs.ExtractEntities(bg, text, "chunk-1")
	if err != nil {
		log.Fatalf("extracting entities: %v", err)
	}
	relations, err := gs.ExtractRelations(bg, text, "chunk-1")
	if err != nil {
		log.Fatalf("extracting relations: %v", err)
	}
	fmt.Printf("extracted %d entities, %d relations\n", len(entities), len(relations))

	// Mirror the extraction into a KnowledgeGraph, which the reasoning
	// engine traverses.
	g := graph.NewKnowledgeGraph()
	for _, e := range entities {
		g.AddEntity(e)
	}
	for _, r := range relations {
		g.AddRelation(r)
	}

	if path := gs.FindPath("alice", "stanford"); path != nil {
		fmt.Printf("path alice -> stanford: %s\n", path.String())
	}

	engine := reasoning.NewEngine(g, reasoning.Config{MaxDepth: 3})
	inferred := engine.InferRelations()
	fmt.Printf("inferred %d new relations, e.g.\n", len(inferred))
	for i, rel := range inferred {
		if i >= 2 {
			break
		}
		fmt.Printf("  %s -[%s]-> %s (confidence %.2f)\n", rel.From, rel.Type, rel.To, rel.Confidence)
	}
	fmt.Println()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
