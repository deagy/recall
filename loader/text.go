package loader

import (
	"context"
	"fmt"
	"os"
)

// TextLoader loads a plain text file as a single document.
type TextLoader struct {
	// MaxBytes caps the number of bytes read; 0 means no cap.
	MaxBytes int64
}

// Load reads the file at path. An empty file is an error because there is
// nothing to embed.
func (l *TextLoader) Load(ctx context.Context, path string) ([]*Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("loader: read %s: %w", path, err)
	}
	if l.MaxBytes > 0 && int64(len(data)) > l.MaxBytes {
		data = data[:l.MaxBytes]
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("loader: %s is empty", path)
	}
	doc := NewDocument(path, baseName(path), path, string(data))
	return []*Document{doc}, nil
}
