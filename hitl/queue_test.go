package hitl

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReviewQueue_EnqueueAndNext_Priority(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("low", "score 0.2", 0.2)
	q.Enqueue("high", "score 0.9", 0.9)
	q.Enqueue("mid", "score 0.5", 0.5)
	assert.Equal(t, 3, q.Count())

	first, ok := q.Next()
	require.True(t, ok)
	assert.Equal(t, "high", first.ChunkID)
	second, _ := q.Next()
	assert.Equal(t, "mid", second.ChunkID)
	third, _ := q.Next()
	assert.Equal(t, "low", third.ChunkID)
	_, ok = q.Next()
	assert.False(t, ok, "empty queue")
}

func TestReviewQueue_Dedup_PendingUpdates(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("c1", "reason A", 0.3)
	q.Enqueue("c1", "reason B", 0.8) // refresh while pending
	assert.Equal(t, 1, q.Count())
	item, _ := q.Next()
	assert.Equal(t, "reason B", item.Reason)
	assert.InDelta(t, 0.8, item.Score, 1e-9)
}

func TestReviewQueue_ReviewedNotReEnqueued(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("c1", "r", 0.5)
	q.MarkReviewed("c1", false)
	assert.Equal(t, 0, q.Count())
	q.Enqueue("c1", "again", 0.9) // should be a no-op
	assert.Equal(t, 0, q.Count())
	assert.Equal(t, 1, q.RejectedCount())
}

func TestReviewQueue_MarkReviewed_Counts(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("a", "", 1.0)
	q.Enqueue("b", "", 0.8)
	q.Enqueue("c", "", 0.6)
	q.MarkReviewed("a", true)
	q.MarkReviewed("b", true)
	q.MarkReviewed("c", false)
	assert.Equal(t, 2, q.ApprovedCount())
	assert.Equal(t, 1, q.RejectedCount())
	assert.Equal(t, 0, q.Count(), "reviewed items are no longer pending")
}

func TestReviewQueue_Pending_Sorted(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("x", "", 0.1)
	q.Enqueue("y", "", 0.9)
	q.Enqueue("z", "", 0.5)
	pending := q.Pending()
	require.Len(t, pending, 3)
	assert.Equal(t, []string{"y", "z", "x"}, []string{pending[0].ChunkID, pending[1].ChunkID, pending[2].ChunkID})
}

func TestReviewQueue_Status(t *testing.T) {
	q := NewReviewQueue()
	q.Enqueue("c1", "", 0.5)
	st, ok := q.Status("c1")
	require.True(t, ok)
	assert.Equal(t, StatusPending, st)
	q.MarkReviewed("c1", true)
	st, _ = q.Status("c1")
	assert.Equal(t, StatusApproved, st)
	_, ok = q.Status("missing")
	assert.False(t, ok)
}

func TestReviewStatus_String(t *testing.T) {
	assert.Equal(t, "pending", StatusPending.String())
	assert.Equal(t, "approved", StatusApproved.String())
	assert.Equal(t, "rejected", StatusRejected.String())
	assert.Equal(t, "unknown", ReviewStatus(99).String())
}
