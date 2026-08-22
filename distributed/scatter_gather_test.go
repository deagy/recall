package distributed

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/deagy/recall/index"
)

// capturedCtx is what a test search function reports about the context the
// fan-out handed it.
type capturedCtx struct {
	deadline time.Time
	hasDL    bool
}

// blockingSearch is a per-shard search func that records the context's
// deadline and then "blocks" until that context is done. It simulates a
// shard that would run longer than the configured Timeout.
func blockingSearch(send chan<- capturedCtx) func(context.Context, *Shard) ([]index.SearchResult, error) {
	return func(ctx context.Context, s *Shard) ([]index.SearchResult, error) {
		deadline, ok := ctx.Deadline()
		select {
		case send <- capturedCtx{deadline: deadline, hasDL: ok}:
		default:
		}
		<-ctx.Done()
		return nil, ctx.Err()
	}
}

// TestScatterGather_TimeoutApplied verifies that ScatterGatherConfig.Timeout
// is actually enforced: the fan-out context carries a matching deadline, a
// slow shard is cut off at the timeout, and the gather returns with the
// shard's error instead of stalling.
func TestScatterGather_TimeoutApplied(t *testing.T) {
	_, sm := newTestShardCluster(1)
	if _, err := sm.CreateShardWithID(fixedNodeID(0), "shard-slow"); err != nil {
		t.Fatal(err)
	}

	send := make(chan capturedCtx, 1)
	start := time.Now()
	_, err := scatterGather(context.Background(), sm,
		&ScatterGatherConfig{Timeout: 50, TotalResults: 10},
		blockingSearch(send))

	cap, ok := <-send
	if !ok {
		t.Fatal("per-shard search was never invoked")
	}
	if !cap.hasDL {
		t.Fatal("fan-out context must carry the configured timeout deadline")
	}
	if until := cap.deadline.Sub(start); until < 30*time.Millisecond || until > 150*time.Millisecond {
		t.Errorf("deadline is %v after start, expected ~50ms (configured Timeout)", until)
	}
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond {
		t.Errorf("gather returned after %v; a slow shard must run until the timeout, then be cut off", elapsed)
	}
	if elapsed > 5*time.Second {
		t.Errorf("gather took %v; the timeout must bound the wait", elapsed)
	}
	if err == nil {
		t.Error("expected the timed-out shard's error to surface")
	}
}

// TestScatterGather_EarlierDeadlineWins verifies that a caller deadline
// closer than the configured Timeout is respected, not extended.
func TestScatterGather_EarlierDeadlineWins(t *testing.T) {
	_, sm := newTestShardCluster(1)
	if _, err := sm.CreateShardWithID(fixedNodeID(0), "shard-ed"); err != nil {
		t.Fatal(err)
	}

	parent, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	send := make(chan capturedCtx, 1)
	start := time.Now()
	_, err := scatterGather(parent, sm,
		&ScatterGatherConfig{Timeout: 60_000, TotalResults: 10},
		blockingSearch(send))
	if err == nil {
		t.Fatal("expected the parent deadline to end the gather")
	}

	cap, ok := <-send
	if !ok || !cap.hasDL {
		t.Fatalf("expected the parent deadline on the fan-out context (cap=%+v ok=%v)", cap, ok)
	}
	if cap.deadline.After(start.Add(30 * time.Millisecond)) {
		t.Errorf("derived deadline %v is later than the parent's ~10ms deadline", cap.deadline.Sub(start))
	}
}

// TestScatterGather_ZeroTimeoutNoDeadline verifies that Timeout=0 (the
// "no timeout" value) adds no deadline to the fan-out context.
func TestScatterGather_ZeroTimeoutNoDeadline(t *testing.T) {
	_, sm := newTestShardCluster(1)
	if _, err := sm.CreateShardWithID(fixedNodeID(0), "shard-nt"); err != nil {
		t.Fatal(err)
	}

	var hasDL bool
	_, err := scatterGather(context.Background(), sm,
		&ScatterGatherConfig{Timeout: 0, TotalResults: 10},
		func(ctx context.Context, s *Shard) ([]index.SearchResult, error) {
			_, hasDL = ctx.Deadline()
			return []index.SearchResult{}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if hasDL {
		t.Error("Timeout=0 must not add a deadline")
	}
}

// TestScatterGather_MaxResultsPerShard verifies the per-shard result cap is
// applied before the merged total is limited.
func TestScatterGather_MaxResultsPerShard(t *testing.T) {
	_, sm := newTestShardCluster(2)
	for i := 0; i < 5; i++ {
		// IDs share an 8-char prefix so each family lands in one shard.
		if err := sm.StoreChunk(context.Background(), chunkWithEmbed(fmt.Sprintf("aaaa-0000-%d", i), "content", 1, 0, 0)); err != nil {
			t.Fatal(err)
		}
		if err := sm.StoreChunk(context.Background(), chunkWithEmbed(fmt.Sprintf("bbbb-0000-%d", i), "content", 1, 0, 0)); err != nil {
			t.Fatal(err)
		}
	}

	cfg := &ScatterGatherConfig{MaxResultsPerShard: 2, TotalResults: 100}
	res, err := ScatterGatherSearch(context.Background(), sm, []float32{1, 0, 0},
		index.SearchOptions{TopK: 10}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 4 {
		t.Fatalf("expected 4 results (2 shards × cap 2), got %d", len(res))
	}
}
