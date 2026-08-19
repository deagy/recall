package hitl

import (
	"sort"
	"sync"
	"time"
)

// ReviewStatus is the lifecycle state of a queued chunk.
type ReviewStatus int

const (
	// StatusPending means the chunk is waiting to be reviewed.
	StatusPending ReviewStatus = iota
	// StatusApproved means a human approved the chunk.
	StatusApproved
	// StatusRejected means a human rejected the chunk.
	StatusRejected
)

// String returns a human-readable status name.
func (s ReviewStatus) String() string {
	switch s {
	case StatusPending:
		return "pending"
	case StatusApproved:
		return "approved"
	case StatusRejected:
		return "rejected"
	default:
		return "unknown"
	}
}

// ReviewItem is a chunk queued for human review.
type ReviewItem struct {
	// ChunkID is the chunk to review.
	ChunkID string

	// Reason explains why the chunk needs review.
	Reason string

	// Score is the priority (e.g., uncertainty); higher is reviewed first.
	Score float64

	// Status is the current lifecycle state.
	Status ReviewStatus

	// EnqueuedAt is when the item entered the queue.
	EnqueuedAt time.Time

	// UpdatedAt is when the item was last changed.
	UpdatedAt time.Time
}

// ReviewQueue is a thread-safe queue of chunks awaiting human review. Items are
// de-duplicated by ChunkID and served in priority order (highest Score first).
type ReviewQueue struct {
	mu    sync.RWMutex
	items map[string]*ReviewItem
}

// NewReviewQueue creates an empty ReviewQueue.
func NewReviewQueue() *ReviewQueue {
	return &ReviewQueue{items: make(map[string]*ReviewItem)}
}

// Enqueue adds a chunk for review. It is idempotent: an existing pending item
// has its reason/score refreshed; an already-reviewed item is left unchanged.
func (q *ReviewQueue) Enqueue(chunkID, reason string, score float64) {
	if chunkID == "" {
		return
	}
	now := time.Now().UTC()
	q.mu.Lock()
	defer q.mu.Unlock()
	if item, ok := q.items[chunkID]; ok {
		if item.Status == StatusPending {
			item.Reason = reason
			item.Score = score
			item.UpdatedAt = now
		}
		return
	}
	q.items[chunkID] = &ReviewItem{
		ChunkID:    chunkID,
		Reason:     reason,
		Score:      score,
		Status:     StatusPending,
		EnqueuedAt: now,
		UpdatedAt:  now,
	}
}

// Next returns and removes the highest-priority pending item, marking it as
// approved or rejected is the caller's job via MarkReviewed. If the queue has no
// pending items, ok is false.
func (q *ReviewQueue) Next() (*ReviewItem, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	var best *ReviewItem
	for _, item := range q.items {
		if item.Status != StatusPending {
			continue
		}
		if best == nil || item.Score > best.Score ||
			(item.Score == best.Score && item.EnqueuedAt.Before(best.EnqueuedAt)) {
			best = item
		}
	}
	if best == nil {
		return nil, false
	}
	delete(q.items, best.ChunkID)
	return best, true
}

// MarkReviewed records a human decision for a chunk and removes it from the
// pending queue.
func (q *ReviewQueue) MarkReviewed(chunkID string, approve bool) {
	status := StatusApproved
	if !approve {
		status = StatusRejected
	}
	now := time.Now().UTC()
	q.mu.Lock()
	defer q.mu.Unlock()
	item, ok := q.items[chunkID]
	if !ok {
		// Record the decision even if it was already popped by Next.
		item = &ReviewItem{ChunkID: chunkID, EnqueuedAt: now}
	}
	item.Status = status
	item.UpdatedAt = now
	q.items[chunkID] = item
}

// Status returns the current status of a chunk, if it is or was queued.
func (q *ReviewQueue) Status(chunkID string) (ReviewStatus, bool) {
	q.mu.RLock()
	defer q.mu.RUnlock()
	item, ok := q.items[chunkID]
	if !ok {
		return StatusPending, false
	}
	return item.Status, true
}

// Pending returns all pending items sorted by descending score (ties broken by
// enqueue time, oldest first).
func (q *ReviewQueue) Pending() []*ReviewItem {
	q.mu.RLock()
	defer q.mu.RUnlock()
	var out []*ReviewItem
	for _, item := range q.items {
		if item.Status == StatusPending {
			out = append(out, item)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].EnqueuedAt.Before(out[j].EnqueuedAt)
	})
	return out
}

// Count returns the number of pending items.
func (q *ReviewQueue) Count() int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	n := 0
	for _, item := range q.items {
		if item.Status == StatusPending {
			n++
		}
	}
	return n
}

// ApprovedCount returns the number of approved items.
func (q *ReviewQueue) ApprovedCount() int { return q.countBy(StatusApproved) }

// RejectedCount returns the number of rejected items.
func (q *ReviewQueue) RejectedCount() int { return q.countBy(StatusRejected) }

func (q *ReviewQueue) countBy(s ReviewStatus) int {
	q.mu.RLock()
	defer q.mu.RUnlock()
	n := 0
	for _, item := range q.items {
		if item.Status == s {
			n++
		}
	}
	return n
}
