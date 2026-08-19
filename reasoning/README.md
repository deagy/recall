# reasoning

Multi-hop reasoning over a `graph.KnowledgeGraph`: pluggable inference
rules, depth-limited path exploration, and confidence propagation.

## Engine

```go
engine := reasoning.NewEngine(g, reasoning.Config{
    MaxDepth:      3,     // bound on path exploration
    MinConfidence: 0.3,   // drop inferred relations below this
    Rules:         reasoning.DefaultRules(),
})
inferred := engine.InferRelations()  // []*InferredRelation{From, To, Type, Confidence, Path}
```

## Inference rules

`InferenceRule` is `Name() string` + `Apply(rel) (*Relation, bool)`:

| Rule | Infers |
|------|--------|
| `TransitiveRule` | a→b, b→c ⇒ a→c (for transitive types) |
| `SymmetricRule` | a→b ⇒ b→a (e.g. knows) |
| `AntiSymmetricRule` | a→b blocks b→a for ordered types |
| `InverseRule` | named inverse relations (works_at → works_for) |
| `CompositionRule` | located_in + works_at → works_in_location |
| `CommonInterestRule` | shared relations → common interest |
| `HierarchyRule` | is_a / part_of hierarchy closure |

## Confidence aggregation

`Aggregator` combines per-edge confidences along a path:
`ProductAggregator` (default, with decay), `MinAggregator`,
`AverageAggregator`; `DefaultAggregator()`.

## Query processing

`EntityExtractor` maps natural-language queries to graph entities
(pattern-based person/location/organization recognition + a small synonym
expansion table), so a question can be turned into a traversal.
