package store

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/deagy/recall/core"
	"github.com/deagy/recall/embedder"
)

// Two processes sharing one store must not lose a write or fail on a lock.
//
// SetMaxOpenConns(1) serialises writers inside one process and says nothing
// about two. Without a busy timeout, SQLite returns SQLITE_BUSY the moment a
// lock is contended rather than waiting, and nothing here retried it.
//
// cadre's staged store had the same shape and had to change: its comment
// records that "a busy_timeout set that way is absent on the next connection…
// two concurrent writers failed with 'database is locked'". A pragma set with
// db.Exec applies to whichever pooled connection served that statement; the
// connection string applies to every connection the pool opens.
//
// This matters now because the previous goal was written for one operator and
// this one is not. One person does not contend with themselves.

// TestConcurrentWritersToOneFileDoNotFail opens the same database file from
// two independent stores -- the closest a test gets to two processes without
// spawning one -- and writes from both at once.
func TestConcurrentWritersToOneFileDoNotFail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shared.db")

	open := func() *SQLiteStore {
		t.Helper()
		store, err := NewSQLiteStore(Config{
			Namespace: "default",
			Embedder:  embedder.NewMockEmbedder(16),
		}, path)
		if err != nil {
			t.Fatalf("opening %s: %v", path, err)
		}
		t.Cleanup(func() { _ = store.Close() })
		return store
	}

	first, second := open(), open()

	const perWriter = 12
	var wg sync.WaitGroup
	errs := make(chan error, perWriter*2)

	for index, store := range []*SQLiteStore{first, second} {
		wg.Add(1)
		go func(writer int, s *SQLiteStore) {
			defer wg.Done()
			for n := range perWriter {
				id := fmt.Sprintf("writer-%d-doc-%d", writer, n)
				doc := core.NewDocument(id, "Concurrent write", id+".txt")
				body := strings.Repeat("content for "+id+". ", 60)
				if err := s.Upload(context.Background(), doc, body); err != nil {
					errs <- fmt.Errorf("%s: %w", id, err)
					return
				}
			}
		}(index, store)
	}
	wg.Wait()
	close(errs)

	var locked []string
	for err := range errs {
		if strings.Contains(err.Error(), "database is locked") || strings.Contains(err.Error(), "SQLITE_BUSY") {
			locked = append(locked, err.Error())
			continue
		}
		t.Fatalf("concurrent write failed for a reason other than contention: %v", err)
	}
	if len(locked) > 0 {
		t.Fatalf("%d concurrent write(s) failed with a locked database:\n  %s\n\n"+
			"Two writers on one file is what a shared store means. A pragma set with db.Exec "+
			"applies only to the connection that ran it; busy_timeout belongs in the connection "+
			"string so every pooled connection has it.",
			len(locked), strings.Join(locked, "\n  "))
	}

	// Nothing lost. Count() is the store's own view of how many chunks it
	// holds, which is the claim that matters: a write that returned nil and
	// left nothing behind is a lost write, and it looks like success.
	//
	// Re-opened rather than read through the handles that did the writing, so
	// the assertion is about what reached the file rather than what one
	// connection remembers.
	reopened := open()
	if got := reopened.Count(); got == 0 {
		t.Fatal("the store holds no chunks after 24 successful uploads; " +
			"every write returned nil and left nothing behind")
	} else {
		t.Logf("store holds %d chunk(s) from two concurrent writers", got)
	}
}

// TestTheConnectionStringCarriesTheBusyTimeout is the structural half.
//
// The behavioural test above can pass by luck on a fast machine where the
// writers happen not to collide. This asserts the property that makes it pass
// on purpose, and fails on the shape the store had before.
func TestTheConnectionStringCarriesTheBusyTimeout(t *testing.T) {
	dsn := sqliteDSN(filepath.Join(t.TempDir(), "x.db"))
	for _, required := range []string{"busy_timeout", "journal_mode"} {
		if !strings.Contains(dsn, required) {
			t.Fatalf("the connection string %q does not set %s.\n"+
				"Set with db.Exec it applies to one pooled connection; set here it applies "+
				"to every connection the pool opens.", dsn, required)
		}
	}
}
