// Package main demonstrates common usage patterns for the recall library.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
	"github.com/deagy/recall/graph"
	"github.com/deagy/recall/index"
	"github.com/deagy/recall/store"
)

func main() {
	ctx := context.Background()

	// Example 1: Basic upload and search
	exampleUploadAndSearch(ctx)

	// Example 2: Hybrid search
	exampleHybridSearch(ctx)

	// Example 3: Knowledge graph
	exampleKnowledgeGraph()

	// Example 4: Graph-based RAG
	exampleGraphRAG()
}

func exampleUploadAndSearch(ctx context.Context) {
	fmt.Println("=== Example 1: Upload and Search ===")

	// Create a memory store
	cfg := store.Config{
		Namespace: "examples",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	memStore, err := store.NewMemoryStore(cfg)
	if err != nil {
		log.Fatalf("failed to create store: %v", err)
	}

	// Upload documents
	docs := []*core.Document{
		core.NewDocument("doc1", "Go Programming", "https://golang.org"),
		core.NewDocument("doc2", "Python Programming", "https://python.org"),
	}

	doc1Content := "Go is a statically typed, compiled programming language designed at Google. It is syntactically similar to C but with memory safety and garbage collection."
	doc2Content := "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."

	for i, doc := range docs {
		content := doc1Content
		if i == 1 {
			content = doc2Content
		}
		if err := memStore.Upload(ctx, doc, content); err != nil {
			log.Fatalf("upload failed: %v", err)
		}
	}
	fmt.Printf("Uploaded %d documents\n", len(docs))

	// Search
	results, err := memStore.Search(ctx, "programming language", index.SearchOptions{TopK: 5})
	if err != nil {
		log.Fatalf("search failed: %v", err)
	}

	fmt.Printf("Search results for 'programming language':\n")
	for i, r := range results {
		content := r.Chunk.Content
		if len(content) > 50 {
			content = content[:50] + "..."
		}
		fmt.Printf("  %d. Score: %.4f - %s\n", i+1, r.Score, content)
	}
	fmt.Println()
}

func exampleHybridSearch(ctx context.Context) {
	fmt.Println("=== Example 2: Hybrid Search ===")

	cfg := store.Config{
		Namespace: "examples",
		Embedder:  embedder.NewMockEmbedder(384),
	}
	memStore, err := store.NewMemoryStore(cfg)
	if err != nil {
		log.Fatalf("failed to create store: %v", err)
	}

	docs := []*core.Document{
		core.NewDocument("doc1", "ML Basics", ""),
		core.NewDocument("doc2", "Deep Learning", ""),
		core.NewDocument("doc3", "NLP", ""),
	}

	contents := []string{
		"Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
		"Deep learning uses neural networks with many layers to model complex patterns in data.",
		"Natural language processing deals with the interaction between computers and human language.",
	}

	for i, doc := range docs {
		if err := memStore.Upload(ctx, doc, contents[i]); err != nil {
			log.Fatalf("hybrid upload failed: %v", err)
		}
	}

	// Hybrid search with vector + BM25 fusion
	results, err := memStore.SearchHybrid(ctx, "neural networks AI", index.SearchOptions{TopK: 5})
	if err != nil {
		log.Fatalf("hybrid search failed: %v", err)
	}

	fmt.Printf("Hybrid search results for 'neural networks AI':\n")
	for i, r := range results {
		content := r.Chunk.Content
		if len(content) > 50 {
			content = content[:50] + "..."
		}
		fmt.Printf("  %d. Score: %.4f - %s\n", i+1, r.Score, content)
	}
	fmt.Println()
}

func exampleKnowledgeGraph() {
	fmt.Println("=== Example 3: Knowledge Graph ===")

	// Create a knowledge graph
	g := graph.NewKnowledgeGraph()

	// Add entities
	g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("bob", "Bob", graph.EntityPerson))
	g.AddEntity(graph.NewEntity("google", "Google", graph.EntityOrganizer))
	g.AddEntity(graph.NewEntity("stanford", "Stanford", graph.EntityOrganizer))

	// Add relations
	g.AddRelation(graph.NewRelation("alice", "bob", "knows", 0.9))
	g.AddRelation(graph.NewRelation("alice", "google", "works_at", 0.8))
	g.AddRelation(graph.NewRelation("bob", "stanford", "studied_at", 0.7))

	fmt.Printf("Graph: %d entities, %d relations\n", g.Count(), g.RelationCount())

	// Find neighbors
	if alice, ok := g.GetEntity("alice"); ok {
		neighbors := g.Neighbors("alice")
		fmt.Printf("Found entity: %s\n", alice.Label)
		fmt.Printf("Alice's neighbors: ")
		for i, n := range neighbors {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Print(n.Label)
		}
		fmt.Println()
	}

	// Find path
	path := g.FindPath("alice", "stanford")
	if path != nil {
		fmt.Printf("Path from Alice to Stanford: %s\n", path.String())
	}

	// Find entities by type
	people := g.FindEntitiesByType(graph.EntityPerson)
	fmt.Printf("People: ")
	for i, p := range people {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(p.Label)
	}
	fmt.Println()
	fmt.Println()
}

func exampleGraphRAG() {
	fmt.Println("=== Example 4: Graph-based RAG ===")

	// Create a graph store
	memStore := store.NewMemoryGraphStore()

	// Extract entities and relations from text
	text := "Alice works at Google. Bob studied at Stanford. Alice knows Bob."

	entities, err := memStore.ExtractEntities(context.Background(), text, "chunk1")
	if err != nil {
		log.Fatalf("extract entities failed: %v", err)
	}
	relations, err := memStore.ExtractRelations(context.Background(), text, "chunk1")
	if err != nil {
		log.Fatalf("extract relations failed: %v", err)
	}

	fmt.Printf("Extracted %d entities and %d relations\n", len(entities), len(relations))

	// Query the graph
	if alice, ok := memStore.GetEntity("alice"); ok {
		fmt.Printf("Found entity: %s (type: %s)\n", alice.Label, alice.Type)
	}

	// Find relations
	rels := memStore.Relations()
	fmt.Printf("All relations: ")
	for i, r := range rels {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%s -[%s]-> %s", r.From, r.Type, r.To)
	}
	fmt.Println()
	fmt.Println()
}
