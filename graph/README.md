# graph

Knowledge graphs: entities, typed/weighted relations, and the traversal and
inference operations on them.

## Core

```go
g := graph.NewKnowledgeGraph()
g.AddEntity(graph.NewEntity("alice", "Alice", graph.EntityPerson))
g.AddRelation(graph.NewRelation("alice", "google", "works_at", 0.9))

g.Neighbors("alice")             // adjacent entities
g.FindPath("alice", "stanford")  // BFS shortest path (*Path)
g.TransitiveClosure("alice")     // all reachable entities
g.CommonNeighbors("alice", "bob")
```

`Entity` carries typed properties (`EntityPerson`, `EntityOrganization`,
...); `Relation` carries type + weight + properties.

## Extraction

- `NERExtractor` interface — bring your own NER model.
- `HeuristicNER` (`NewHeuristicNER()`) — dependency-free heuristic
  entity recognition.
- `PatternRelationExtractor` + `DefaultPatterns()` — regex relation
  patterns (works_at, located_in, founded_by, part_of, ...);
  `NewRelationPattern(name, regex, fromIndex, toIndex)` for custom ones.

## Learning & analytics

- `TransE` — TransE-style embedding training over triples
  (`NewTransE(store)`, `TrainOptions`); `LinkPrediction` ranks missing
  edges; `NearestNeighbors` finds similar entities via a `GraphEmbedder`.
- `EvaluateMetrics` — training/link-prediction evaluation metrics.
- Entity similarity utilities (cosine over entity embeddings) in
  `similarity.go`.

`store` wraps these as `GraphStore` (memory + SQLite persistence).
