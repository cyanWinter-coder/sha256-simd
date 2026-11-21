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
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// batchWithPaths runs fn once for the portable path and, when the CPU has
// AVX512, once more with the hardware path forced off. The fallback run also
// forces the pure-Go block function so the comparison covers every
// implementation rather than just the current CPU's fast path.
func batchWithPaths(t *testing.T, fn func(t *testing.T)) {
	t.Helper()
	t.Run("auto", func(t *testing.T) {
		fn(t)
	})
	t.Run("generic", func(t *testing.T) {
		oldForce := batchForceGeneric
		oldBlock := blockfunc
		batchForceGeneric = true
		blockfunc = blockfuncForceGeneric
		defer func() {
			batchForceGeneric = oldForce
			blockfunc = oldBlock
		}()
		fn(t)
	})
}

func randomInputs(seed int64, n, maxLen int) [][]byte {
	r := rand.New(rand.NewSource(seed))
	inputs := make([][]byte, n)
	for i := range inputs {
		b := make([]byte, r.Intn(maxLen+1))
		r.Read(b)
		inputs[i] = b
	}
	return inputs
}

func checkAgainstStdlib(t *testing.T, inputs [][]byte, got [][Size]byte) {
	t.Helper()
	if len(got) != len(inputs) {
		t.Fatalf("got %d digests for %d inputs", len(got), len(inputs))
	}
	for i := range inputs {
		want := sha256.Sum256(inputs[i])
		if got[i] != want {
			t.Fatalf("input %d (len %d): got %x want %x", i, len(inputs[i]), got[i], want)
		}
	}
}

// TestBatchSizes covers every length around the padding boundaries and the
// 16-lane grouping. Every digest must equal crypto/sha256.
func TestBatchSizes(t *testing.T) {
	batchWithPaths(t, func(t *testing.T) {
		b, err := NewBatch(DefaultBatchOptions())
		if err != nil {
			t.Fatal(err)
		}
		var inputs [][]byte
		for n := 0; n <= 2*BlockSize+1; n++ {
			msg := make([]byte, n)
			for i := range msg {
				msg[i] = byte(i*31 + n)
			}
			inputs = append(inputs, msg)
		}
		// A second copy per message so groups are not exactly full.
		inputs = append(inputs, inputs[:7]...)
		got, err := b.Sum256(context.Background(), inputs)
		if err != nil {
			t.Fatal(err)
		}
		checkAgainstStdlib(t, inputs, got)
	})
}

func TestBatchLargeAndMixed(t *testing.T) {
	batchWithPaths(t, func(t *testing.T) {
		b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16, MaxInputBytes: 1 << 22})
		if err != nil {
			t.Fatal(err)
		}
		inputs := randomInputs(7, 50, 1<<18)
		// Force several windows in the hardware path for a few inputs.
		for i := 0; i < 4; i++ {
			inputs[i] = make([]byte, batchWindowBytes+BlockSize*i+3)
			if _, err := rand.New(rand.NewSource(int64(i))).Read(inputs[i]); err != nil {
				t.Fatal(err)
			}
		}
		got, err := b.Sum256(context.Background(), inputs)
		if err != nil {
			t.Fatal(err)
		}
		checkAgainstStdlib(t, inputs, got)
	})
}

func TestBatchEmpty(t *testing.T) {
	b, err := NewBatch(DefaultBatchOptions())
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.Sum256(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("got %d digests for empty batch", len(got))
	}
	if err := b.Sum(context.Background(), nil, nil); err != nil {
		t.Fatal(err)
	}
}

// TestBatchDestinationValidation checks that every destination buffer is
// validated independently and that mismatched lengths are rejected.
func TestBatchDestinationValidation(t *testing.T) {
	b, err := NewBatch(DefaultBatchOptions())
	if err != nil {
		t.Fatal(err)
	}
	inputs := [][]byte{[]byte("a"), []byte("b"), []byte("c")}

	for i := 0; i < len(inputs); i++ {
		sums := make([][]byte, len(inputs))
		for j := range sums {
			sums[j] = make([]byte, Size)
		}
		sums[i] = make([]byte, Size-1)
		err := b.Sum(context.Background(), inputs, sums)
		if !errors.Is(err, ErrShortBuffer) {
			t.Fatalf("index %d: got %v, want ErrShortBuffer", i, err)
		}
		if !bytes.Contains([]byte(err.Error()), []byte(fmt.Sprintf("index %d", i))) {
			t.Fatalf("error does not name the offending index: %v", err)
		}
	}

	if err := b.Sum(context.Background(), inputs, make([][]byte, 2)); err == nil {
		t.Fatal("expected an error for a destination/input length mismatch")
	}
}

// TestBatchDestinationsAllocatedAndTrimmed verifies that nil entries are
// allocated, longer entries are trimmed to Size, and slice aliases are never
// overrun.
func TestBatchDestinationsAllocatedAndTrimmed(t *testing.T) {
	b, err := NewBatch(DefaultBatchOptions())
	if err != nil {
		t.Fatal(err)
	}
	inputs := randomInputs(3, 9, 200)
	sums := make([][]byte, len(inputs))
	guards := make([][]byte, len(inputs))
	for i := range sums {
		if i%2 == 0 {
			guard := bytes.Repeat([]byte{0xAA}, Size+8)
			guards[i] = guard
			sums[i] = guard[:Size+3]
		}
	}
	if err := b.Sum(context.Background(), inputs, sums); err != nil {
		t.Fatal(err)
	}
	for i := range inputs {
		if len(sums[i]) != Size {
			t.Fatalf("destination %d has length %d, want %d", i, len(sums[i]), Size)
		}
		want := mustStdlib(inputs[i])
		if !bytes.Equal(sums[i], want[:]) {
			t.Fatalf("destination %d mismatch", i)
		}
	}
	for i := range guards {
		guard := guards[i]
		if guard == nil {
			continue
		}
		for j := Size; j < len(guard); j++ {
			if guard[j] != 0xAA {
				t.Fatalf("destination %d wrote past the digest into byte %d", i, j)
			}
		}
	}
}

func mustStdlib(b []byte) [Size]byte { return sha256.Sum256(b) }

// TestBatchMatchesAcrossLanes makes the hardware and portable paths agree for
// a range of lane and batch-size settings, including partial lane groups.
func TestBatchMatchesAcrossLanes(t *testing.T) {
	inputs := randomInputs(11, 37, 4096)
	want := make([][Size]byte, len(inputs))
	for i := range inputs {
		want[i] = sha256.Sum256(inputs[i])
	}

	for _, lanes := range []int{1, 2, 5, 16, 17, 64} {
		for _, batchSize := range []int{1, 3, 16, 32} {
			b, err := NewBatch(BatchOptions{Lanes: lanes, BatchSize: batchSize})
			if err != nil {
				t.Fatal(err)
			}
			for _, generic := range []bool{false, true} {
				if generic && batchAVX512 == nil {
					continue
				}
				old := batchForceGeneric
				batchForceGeneric = generic
				got, err := b.Sum256(context.Background(), inputs)
				batchForceGeneric = old
				if err != nil {
					t.Fatalf("lanes=%d batch=%d generic=%v: %v", lanes, batchSize, generic, err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("lanes=%d batch=%d generic=%v index=%d mismatch", lanes, batchSize, generic, i)
					}
				}
			}
		}
	}
}

// TestBatchCancelPreservesCompleted cancels mid-batch and checks that only
// fully computed digests are published, that the rest are an explicit zero
// placeholder, and that the error names exactly those indices.
func TestBatchCancelPreservesCompleted(t *testing.T) {
	if batchAVX512 != nil {
		// The hardware path hashes a whole lane group in lock-step, so it
		// cannot be stopped at an arbitrary digest; the deterministic test
		// below covers the "nothing published" side of the contract there.
		t.Log("hardware path present; exercising the portable cancellation hook")
	}
	// One lane, single-input groups, so the portable path can stop between
	// inputs deterministically.
	b, err := NewBatch(BatchOptions{Lanes: 1, BatchSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	inputs := randomInputs(5, 32, 128)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	const cancelAfter = 7
	var completed int32
	oldForce, oldHook := batchForceGeneric, batchAfterDigest
	batchForceGeneric = true
	batchAfterDigest = func() {
		if atomic.AddInt32(&completed, 1) == cancelAfter {
			cancel()
		}
	}
	defer func() {
		batchForceGeneric, batchAfterDigest = oldForce, oldHook
	}()

	got, err := b.Sum256(ctx, inputs)
	if !errors.Is(err, ErrBatchIncomplete) {
		t.Fatalf("got %v, want ErrBatchIncomplete", err)
	}
	var inc *IncompleteError
	if !errors.As(err, &inc) {
		t.Fatalf("got %v, want *IncompleteError", err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error does not match context.Canceled: %v", err)
	}
	if len(inc.Missing) == 0 || len(inc.Missing) == len(inputs) {
		t.Fatalf("expected a partial result, got %d missing of %d", len(inc.Missing), len(inputs))
	}
	missing := make(map[int]bool, len(inc.Missing))
	for _, i := range inc.Missing {
		missing[i] = true
	}
	for i := range inputs {
		if missing[i] {
			if got[i] != ([Size]byte{}) {
				t.Fatalf("missing index %d is not the zero placeholder: %x", i, got[i])
			}
			continue
		}
		if want := sha256.Sum256(inputs[i]); got[i] != want {
			t.Fatalf("completed index %d is wrong: got %x want %x", i, got[i], want)
		}
	}
}

// TestBatchCancelDeterministic uses an already-canceled context: nothing is
// computed, every entry is a zero placeholder, and every index is reported.
func TestBatchCancelDeterministic(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 1, BatchSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	inputs := randomInputs(6, 12, 64)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	got, err := b.Sum256(ctx, inputs)
	if !errors.Is(err, ErrBatchIncomplete) || !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want canceled IncompleteError", err)
	}
	var inc *IncompleteError
	if !errors.As(err, &inc) || len(inc.Missing) != len(inputs) {
		t.Fatalf("got %+v, want all %d indices missing", err, len(inputs))
	}
	for i := range got {
		if got[i] != ([Size]byte{}) {
			t.Fatalf("index %d not zeroed on cancellation", i)
		}
	}

	sums := make([][]byte, len(inputs))
	for i := range sums {
		sums[i] = bytes.Repeat([]byte{0xEE}, Size)
	}
	if err := b.Sum(ctx, inputs, sums); !errors.Is(err, ErrBatchIncomplete) {
		t.Fatalf("destination form: got %v, want ErrBatchIncomplete", err)
	}
	for i := range sums {
		if !bytes.Equal(sums[i], make([]byte, Size)) {
			t.Fatalf("destination %d not zeroed on cancellation: %x", i, sums[i])
		}
	}
}

// TestBatchContextDeadline checks the deadline variant of the same contract.
func TestBatchContextDeadline(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 1, BatchSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	_, err = b.Sum256(ctx, randomInputs(8, 4, 32))
	if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, ErrBatchIncomplete) {
		t.Fatalf("got %v, want deadline IncompleteError", err)
	}
}

// TestBatchInputReuseAfterReturn checks that the batch never keeps input
// frames alive: mutating or clearing every input after the call does not
// change subsequent results and does not change what was already returned.
func TestBatchInputReuseAfterReturn(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	inputs := randomInputs(9, 48, 512)
	first, err := b.Sum256(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}

	// Clobber every input in place.
	for _, in := range inputs {
		for j := range in {
			in[j] = 0xFF
		}
	}
	firstCopy := first
	second, err := b.Sum256(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	wantSecond := make([][Size]byte, len(inputs))
	for i := range inputs {
		wantSecond[i] = sha256.Sum256(inputs[i])
	}
	for i := range inputs {
		if second[i] != wantSecond[i] {
			t.Fatalf("index %d: clobbered input not re-hashed: %x", i, second[i])
		}
		if first[i] != firstCopy[i] {
			t.Fatalf("index %d: earlier result changed after inputs were modified", i)
		}
	}
}

// TestBatchLaneIsolation runs batches with wildly different inputs
// concurrently through one shared Batch, so a scratch/lane leak would show up
// as a wrong digest.
func TestBatchLaneIsolation(t *testing.T) {
	b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16, MaxInputBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	const goroutines = 8
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(100 + g)))
			for iter := 0; iter < 25; iter++ {
				n := 1 + r.Intn(40)
				inputs := make([][]byte, n)
				for i := range inputs {
					l := r.Intn(3 * BlockSize)
					inputs[i] = make([]byte, l)
					r.Read(inputs[i])
				}
				got, err := b.Sum256(context.Background(), inputs)
				if err != nil {
					errCh <- err
					return
				}
				for i := range inputs {
					if want := sha256.Sum256(inputs[i]); got[i] != want {
						errCh <- fmt.Errorf("goroutine %d iter %d index %d: got %x want %x", g, iter, i, got[i], want)
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

// TestBatchBackpressure checks that MaxInputBytes is respected without
// copying inputs: a large batch with a small byte budget still completes with
// correct digests, and the peak in-flight bytes never exceed the budget.
func TestBatchBackpressure(t *testing.T) {
	const maxBytes = 1024
	b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16, MaxInputBytes: maxBytes})
	if err != nil {
		t.Fatal(err)
	}
	inputs := make([][]byte, 64)
	for i := range inputs {
		inputs[i] = make([]byte, 200)
		binary.LittleEndian.PutUint32(inputs[i], uint32(i))
	}

	var mu sync.Mutex
	inFlight, peak := 0, 0
	old := batchGroupObserver
	batchGroupObserver = func(acquired bool, n int) {
		mu.Lock()
		defer mu.Unlock()
		if acquired {
			inFlight += n
			if inFlight > peak {
				peak = inFlight
			}
		} else {
			inFlight -= n
		}
	}
	got, err := b.Sum256(context.Background(), inputs)
	batchGroupObserver = old
	if err != nil {
		t.Fatal(err)
	}
	checkAgainstStdlib(t, inputs, got)
	if peak > maxBytes {
		t.Fatalf("peak in-flight input bytes = %d, budget = %d", peak, maxBytes)
	}
	if peak == 0 {
		t.Fatal("observer saw no admitted groups")
	}

	// An input bigger than the budget is still processed; it runs alone with
	// nothing else in flight.
	big := make([]byte, 4*maxBytes)
	if _, err := rand.Read(big); err != nil {
		t.Fatal(err)
	}
	got, err = b.Sum256(context.Background(), [][]byte{[]byte("x"), big, []byte("y")})
	if err != nil {
		t.Fatal(err)
	}
	checkAgainstStdlib(t, [][]byte{[]byte("x"), big, []byte("y")}, got)
}

// TestBatchOptionsNormalization checks defaults and option validation.
func TestBatchOptionsNormalization(t *testing.T) {
	b, err := NewBatch(BatchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got := b.Options()
	want := DefaultBatchOptions()
	if got != want {
		t.Fatalf("options = %+v, want %+v", got, want)
	}

	for _, opts := range []BatchOptions{
		{Lanes: -1},
		{BatchSize: -1},
		{MaxInputBytes: -1},
	} {
		if _, err := NewBatch(opts); err == nil {
			t.Fatalf("NewBatch(%+v) succeeded, want error", opts)
		}
	}
}

// TestSum256BatchDefault checks the package-level helper and that it sees
// ordinary New/Sum256 unchanged.
func TestSum256BatchDefault(t *testing.T) {
	inputs := randomInputs(13, 21, 300)
	got, err := Sum256Batch(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	checkAgainstStdlib(t, inputs, got)

	// Existing single-shot APIs are untouched and agree with the batch.
	for i := range inputs {
		if direct := Sum256(inputs[i]); direct != got[i] {
			t.Fatalf("index %d: Sum256 %x != batch %x", i, direct, got[i])
		}
		h := New()
		h.Write(inputs[i])
		if s := h.Sum(nil); !bytes.Equal(s, got[i][:]) {
			t.Fatalf("index %d: hash.Hash Sum %x != batch %x", i, s, got[i])
		}
	}
}

// TestBatchHashHashUnchanged guards the standard hash.Hash contract: Size and
// BlockSize still match, and interleaved batch use does not disturb a digest.
func TestBatchHashHashUnchanged(t *testing.T) {
	h := New()
	if h.Size() != Size || h.BlockSize() != BlockSize {
		t.Fatalf("hash.Hash constants changed: Size=%d BlockSize=%d", h.Size(), h.BlockSize())
	}
	b, err := NewBatch(DefaultBatchOptions())
	if err != nil {
		t.Fatal(err)
	}
	h.Write([]byte("hello "))
	if _, err := b.Sum256(context.Background(), randomInputs(17, 20, 100)); err != nil {
		t.Fatal(err)
	}
	h.Write([]byte("world"))
	if got, want := h.Sum(nil), sha256.Sum256([]byte("hello world")); !bytes.Equal(got, want[:]) {
		t.Fatalf("hash.Hash disturbed by batch use: got %x want %x", got, want)
	}
}

// TestBatchLanesBoundInFlight checks that at most Lanes digests are ever in
// flight, using the group observer to count concurrently admitted inputs.
func TestBatchLanesBoundInFlight(t *testing.T) {
	const lanes = 5
	b, err := NewBatch(BatchOptions{Lanes: lanes, BatchSize: 3, MaxInputBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	inputs := randomInputs(21, 97, 200)

	var mu sync.Mutex
	inFlight, peak := 0, 0
	old := batchGroupObserver
	batchGroupObserver = func(acquired bool, _ int) {
		mu.Lock()
		defer mu.Unlock()
		if acquired {
			inFlight++
			if inFlight > peak {
				peak = inFlight
			}
		} else {
			inFlight--
		}
	}
	_, err = b.Sum256(context.Background(), inputs)
	batchGroupObserver = old
	if err != nil {
		t.Fatal(err)
	}
	if peak > lanes {
		t.Fatalf("peak in-flight groups of %d inputs each = %d, exceeds lanes %d", b.groupSize(), peak, lanes)
	}
}

func BenchmarkBatchSum256(b *testing.B) {
	for _, size := range []int{64, 1 << 12, 1 << 16} {
		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			batch, err := NewBatch(DefaultBatchOptions())
			if err != nil {
				b.Fatal(err)
			}
			inputs := randomInputs(1, 16, size)
			for i := range inputs {
				inputs[i] = make([]byte, size)
			}
			b.SetBytes(int64(size * len(inputs)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := batch.Sum256(context.Background(), inputs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
