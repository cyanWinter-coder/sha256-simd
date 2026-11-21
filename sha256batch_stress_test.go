/*
 * Minio Cloud Storage, (C) 2026 Minio, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package sha256

import (
	"context"
	"crypto/sha256"
	"errors"
	"sync"
	"testing"
	"time"
)

// checkPartialResult enforces the cancellation contract: completed digests
// match crypto/sha256, missing ones are the zero placeholder, and a message is
// never partially published.
func checkPartialResult(t *testing.T, inputs [][]byte, got [][Size]byte, err error) {
	t.Helper()
	missing := make(map[int]bool)
	if err != nil {
		var inc *IncompleteError
		if !errors.As(err, &inc) {
			t.Fatalf("got %v, want *IncompleteError", err)
		}
		if !errors.Is(err, ErrBatchIncomplete) {
			t.Fatalf("error does not match ErrBatchIncomplete: %v", err)
		}
		for _, i := range inc.Missing {
			if i < 0 || i >= len(inputs) {
				t.Fatalf("missing index %d out of range", i)
			}
			missing[i] = true
			if got[i] != ([Size]byte{}) {
				t.Fatalf("index %d: non-zero placeholder for a missing digest", i)
			}
		}
	}
	for i := range inputs {
		if missing[i] {
			continue
		}
		if want := sha256.Sum256(inputs[i]); got[i] != want {
			t.Fatalf("index %d (len %d): %x != %x", i, len(inputs[i]), got[i], want)
		}
	}
}

// TestBatchStressCancelDeadline runs many batches with deadlines that land at
// random points, to shake out races between completion, cancellation and the
// backpressure budget. It must never deadlock and must never publish a wrong
// digest.
func TestBatchStressCancelDeadline(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 4, BatchSize: 3, MaxInputBytes: 1 << 12})
	if err != nil {
		t.Fatal(err)
	}
	for iter := 0; iter < 150; iter++ {
		inputs := randomInputs(int64(iter), 1+iter%20, 512)
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(iter%7)*time.Microsecond)
		got, err := b.Sum256(ctx, inputs)
		cancel()
		checkPartialResult(t, inputs, got, err)
	}
}

// TestBatchConcurrentCancel hammers one shared Batch from many goroutines,
// half of which cancel almost immediately. It verifies the API stays
// race-free and that shared lane scratch never leaks between callers.
func TestBatchConcurrentCancel(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16, MaxInputBytes: 1 << 16})
	if err != nil {
		t.Fatal(err)
	}
	const goroutines = 12
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for iter := 0; iter < 40; iter++ {
				inputs := randomInputs(int64(g*1000+iter), 1+iter%17, 1024)
				ctx := context.Background()
				cancel := func() {}
				if g%2 == 0 {
					ctx, cancel = context.WithTimeout(ctx, time.Duration(iter%3)*time.Microsecond)
				}
				got, err := b.Sum256(ctx, inputs)
				cancel()
				var inc *IncompleteError
				if err != nil {
					if !errors.As(err, &inc) {
						errCh <- err
						return
					}
				}
				for i := range inputs {
					if inc != nil && isMissing(inc, i) {
						continue
					}
					if want := sha256.Sum256(inputs[i]); got[i] != want {
						errCh <- errors.New("digest mismatch under concurrency")
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

func isMissing(inc *IncompleteError, i int) bool {
	for _, m := range inc.Missing {
		if m == i {
			return true
		}
	}
	return false
}
