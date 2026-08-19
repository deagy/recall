package config

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// Watcher polls a configuration file for changes and invokes a callback
// with the reloaded configuration when the file's modification time or
// size changes. Invalid files are skipped (and remembered via LastError)
// so a bad edit never takes a running service down.
//
// Watcher is safe for concurrent use.
type Watcher struct {
	path     string
	interval time.Duration
	onChange func(*Config) error

	mu       sync.Mutex
	lastMod  time.Time
	lastSize int64
	lastErr  error
	errSet   bool

	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
	running  bool
}

// NewWatcher creates a Watcher for path, polling every interval (minimum
// 100ms). onChange receives the new validated configuration; returning an
// error from onChange rejects the reload without changing the reported
// last error. The current on-disk state is recorded at creation so the
// first poll does not fire.
func NewWatcher(path string, interval time.Duration, onChange func(*Config) error) (*Watcher, error) {
	if path == "" {
		return nil, fmt.Errorf("watcher: path is required")
	}
	if onChange == nil {
		return nil, fmt.Errorf("watcher: onChange callback is required")
	}
	if interval < 100*time.Millisecond {
		interval = 100 * time.Millisecond
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("watcher stat %s: %w", path, err)
	}
	return &Watcher{
		path:     path,
		interval: interval,
		onChange: onChange,
		lastMod:  info.ModTime(),
		lastSize: info.Size(),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}, nil
}

// Start begins the background polling goroutine. It returns an error if
// the watcher is already running.
func (w *Watcher) Start() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.running {
		return fmt.Errorf("watcher: already running")
	}
	w.running = true
	go w.poll()
	return nil
}

// Stop terminates the polling goroutine and waits for it to exit. It is
// a no-op when the watcher was never started (or already stopped).
func (w *Watcher) Stop() {
	w.mu.Lock()
	if !w.running {
		w.mu.Unlock()
		return
	}
	w.stopOnce.Do(func() { close(w.stop) })
	w.mu.Unlock()
	<-w.done
}

// LastError returns the most recent reload failure, if any.
func (w *Watcher) LastError() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.errSet {
		return nil
	}
	return w.lastErr
}

// Path returns the watched file path.
func (w *Watcher) Path() string { return w.path }

func (w *Watcher) poll() {
	defer func() {
		w.mu.Lock()
		w.running = false
		w.mu.Unlock()
		close(w.done)
	}()
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	for {
		select {
		case <-w.stop:
			return
		case <-ticker.C:
			w.check()
		}
	}
}

func (w *Watcher) check() {
	info, err := os.Stat(w.path)
	if err != nil {
		w.setErr(fmt.Errorf("watcher stat %s: %w", w.path, err))
		return
	}
	w.mu.Lock()
	changed := !info.ModTime().Equal(w.lastMod) || info.Size() != w.lastSize
	if changed {
		w.lastMod = info.ModTime()
		w.lastSize = info.Size()
	}
	w.mu.Unlock()
	if !changed {
		return
	}

	cfg, err := Load(w.path)
	if err != nil {
		w.setErr(err)
		return
	}
	if err := w.onChange(cfg); err != nil {
		w.setErr(fmt.Errorf("watcher applying config: %w", err))
		return
	}
	w.setErr(nil)
}

func (w *Watcher) setErr(err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err == nil {
		w.errSet = false
		w.lastErr = nil
		return
	}
	w.errSet = true
	w.lastErr = err
}
