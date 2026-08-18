package store

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/index"
	"github.com/stretchr/testify/require"
)

// TestSQLiteStore_ConcurrentUploadAndSearch uploads documents in parallel while
// other goroutines continuously search, crossing the HNSW threshold so the
// in-memory mirror is built and extended under lock. Without proper locking the
// Go runtime detects the concurrent map access and the test fails fatally.
func TestSQLiteStore_ConcurrentUploadAndSearch(t *testing.T) {
	s := newTestSQLiteStore(t)
	ctx := context.Background()

	const writers = 4
	const perWriter = 300 // 1200 total: crosses index.HNSWThreshold

	var wg sync.WaitGroup
	errCh := make(chan error, writers+1)

	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				id := fmt.Sprintf("conc-w%d-d%d", w, i)
				doc := core.NewDocument(id, id, "t.txt")
				content := fmt.Sprintf("concurrency document number %03d for writer %d with padding to exceed the minimum chunk size", i, w)
				if err := s.Upload(ctx, doc, content); err != nil {
					errCh <- fmt.Errorf("upload %s: %w", id, err)
					return
				}
			}
		}(w)
	}

	// Reader hammers Search while uploads mutate the mirror.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 300; i++ {
			if _, err := s.Search(ctx, "concurrency document", index.SearchOptions{TopK: 5}); err != nil {
				errCh <- fmt.Errorf("search %d: %w", i, err)
				return
			}
		}
	}()

	wg.Wait()
	close(errCh)
	for err := range errCh {
		require.NoError(t, err, "concurrent store operations must not fail")
	}

	require.Equal(t, writers*perWriter, s.Count(), "all uploaded chunks should be persisted")
}
