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
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// Default values used by BatchOptions when a field is left at zero.
const (
	// DefaultBatchLanes is the number of digest lanes a Batch processes
	// concurrently. It matches the width of the AVX512 multi-buffer
	// implementation, so a full batch keeps every hardware lane busy.
	DefaultBatchLanes = 16

	// DefaultBatchSize is the largest number of inputs handed to a single
	// batch step.
	DefaultBatchSize = 16

	// DefaultBatchMaxInputBytes bounds how many input bytes a Batch
	// references (and therefore keeps pinned) at any one time.
	DefaultBatchMaxInputBytes = 8 << 20

	// batchWindowBytes bounds how many bytes one lane contributes to a
	// single SIMD call. Larger messages are fed in windows so a canceled
	// context is noticed promptly; the state is carried between windows, so
	// the digest is identical to hashing the whole message at once.
	batchWindowBytes = 1 << 18 // 256 KiB per lane
)

// Errors returned by the batch API.
var (
	// ErrShortBuffer is returned by Batch.Sum when a caller supplied
	// destination entry cannot hold a full digest.
	ErrShortBuffer = errors.New("sha256: batch destination buffer too short")

	// ErrBatchIncomplete is reported through IncompleteError for any digest
	// that was not computed. It is always matched by an IncompleteError, so
	// errors.Is(err, ErrBatchIncomplete) works even when the underlying
	// cause is a context cancellation.
	ErrBatchIncomplete = errors.New("sha256: batch incomplete")
)

// BatchOptions configures a Batch.
//
// All fields are optional; zero values are replaced by the matching
// Default* constant. Negative values are rejected by NewBatch.
type BatchOptions struct {
	// MaxInputBytes bounds the total length of the inputs a Batch may have
	// in flight at any one time. Inputs are never copied; this is a
	// backpressure limit on how much caller memory the batch keeps pinned
	// while work is queued and in progress. When the next batch would
	// exceed the limit, the caller waits for in-flight work to drain
	// instead of queuing (or copying) the whole input set. A single input
	// larger than the limit is still processed on its own.
	MaxInputBytes int

	// Lanes bounds the number of digests processed concurrently. The batch
	// is split into groups of at most 16 inputs (the AVX512 lane width)
	// and no more than Lanes digests are in flight. Worker parallelism is
	// capped at runtime.GOMAXPROCS.
	Lanes int

	// BatchSize bounds the number of inputs handed to one batch step. It is
	// capped by the SIMD lane width (16) for the hardware path and by Lanes
	// for the total number of in-flight digests.
	BatchSize int
}

// DefaultBatchOptions returns BatchOptions filled with the package defaults.
func DefaultBatchOptions() BatchOptions {
	return BatchOptions{
		MaxInputBytes: DefaultBatchMaxInputBytes,
		Lanes:         DefaultBatchLanes,
		BatchSize:     DefaultBatchSize,
	}
}

// IncompleteError reports that some digests of a batch were not computed.
//
// The returned digest slice (or caller supplied destination) still contains
// every digest that did complete. Entries listed in Missing are left as an
// explicit all-zero placeholder; a partially computed message is never
// published.
type IncompleteError struct {
	// Err is the cause, normally context.Canceled or context.DeadlineExceeded.
	Err error

	// Missing contains the ascending indices of digests that were not
	// computed.
	Missing []int

	// Total is the number of inputs in the batch.
	Total int
}

func (e *IncompleteError) Error() string {
	return fmt.Sprintf("sha256: batch incomplete: %d of %d digests not computed: %v", len(e.Missing), e.Total, e.Err)
}

// Unwrap returns the underlying cause so errors.Is/As see it.
func (e *IncompleteError) Unwrap() error { return e.Err }

// Is reports both ErrBatchIncomplete and the wrapped context error.
func (e *IncompleteError) Is(target error) bool {
	if target == ErrBatchIncomplete {
		return true
	}
	return errors.Is(e.Err, target)
}

// Batch computes SHA-256 digests for many independent inputs.
//
// A Batch reuses its internal lane and assembly scratch storage between
// calls. It is safe for concurrent use by multiple goroutines, and it never
// retains references to the input slices after a call returns, so callers may
// reuse or modify their buffers immediately.
type Batch struct {
	opts BatchOptions
	pool sync.Pool
}

// NewBatch creates a Batch. Zero option fields select the package defaults.
func NewBatch(opts BatchOptions) (*Batch, error) {
	if opts.Lanes < 0 {
		return nil, fmt.Errorf("sha256: invalid batch lanes %d", opts.Lanes)
	}
	if opts.BatchSize < 0 {
		return nil, fmt.Errorf("sha256: invalid batch size %d", opts.BatchSize)
	}
	if opts.MaxInputBytes < 0 {
		return nil, fmt.Errorf("sha256: invalid batch input byte limit %d", opts.MaxInputBytes)
	}
	if opts.Lanes == 0 {
		opts.Lanes = DefaultBatchLanes
	}
	if opts.BatchSize == 0 {
		opts.BatchSize = DefaultBatchSize
	}
	if opts.MaxInputBytes == 0 {
		opts.MaxInputBytes = DefaultBatchMaxInputBytes
	}
	return &Batch{opts: opts}, nil
}

// Options returns the normalized options this Batch was created with.
func (b *Batch) Options() BatchOptions { return b.opts }

// Sum256 hashes every input and returns the digests in input order.
//
// Each returned digest is exactly Size bytes. If ctx is canceled the digests
// that already completed are returned together with an *IncompleteError;
// entries that did not complete are all zero. A partially hashed message is
// never returned.
func (b *Batch) Sum256(ctx context.Context, inputs [][]byte) ([][Size]byte, error) {
	res, missing := b.sumContext(ctx, inputs)
	if len(missing) == 0 {
		return res, nil
	}
	return res, newIncompleteError(ctx, len(inputs), missing)
}

// Sum hashes every input and writes digest i into sums[i][:Size].
//
// sums must have exactly one entry per input. A nil entry is allocated with
// Size bytes; any other entry must be at least Size bytes long and is
// validated independently, so a short buffer reports the offending index
// before any hashing takes place. On return every entry has exactly Size
// usable bytes.
//
// On context cancellation completed digests are preserved and the remaining
// entries are zeroed; the returned *IncompleteError lists their indices.
func (b *Batch) Sum(ctx context.Context, inputs [][]byte, sums [][]byte) error {
	if len(sums) != len(inputs) {
		return fmt.Errorf("sha256: batch has %d inputs but %d destination buffers", len(inputs), len(sums))
	}
	for i, sum := range sums {
		if sum != nil && len(sum) < Size {
			return fmt.Errorf("%w: index %d has %d bytes, need %d", ErrShortBuffer, i, len(sum), Size)
		}
	}

	res, missing := b.sumContext(ctx, inputs)

	nextMissing := 0
	for i := range sums {
		if sums[i] == nil {
			sums[i] = make([]byte, Size)
		} else {
			sums[i] = sums[i][:Size]
		}
		if nextMissing < len(missing) && missing[nextMissing] == i {
			for j := range sums[i] {
				sums[i][j] = 0
			}
			nextMissing++
			continue
		}
		copy(sums[i], res[i][:])
	}
	if len(missing) == 0 {
		return nil
	}
	return newIncompleteError(ctx, len(inputs), missing)
}

// Sum256Batch hashes inputs with the default batch configuration.
//
// It shares a process-wide Batch, so internal lanes and assembly scratch are
// reused between calls. Use NewBatch when non-default limits are required.
func Sum256Batch(ctx context.Context, inputs [][]byte) ([][Size]byte, error) {
	return defaultBatch().Sum256(ctx, inputs)
}

var (
	defaultBatchOnce sync.Once
	defaultBatchVal  *Batch
)

func defaultBatch() *Batch {
	defaultBatchOnce.Do(func() {
		// Defaults are always valid; NewBatch cannot fail here.
		defaultBatchVal, _ = NewBatch(DefaultBatchOptions())
	})
	return defaultBatchVal
}

func newIncompleteError(ctx context.Context, total int, missing []int) error {
	err := ErrBatchIncomplete
	if ctx != nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	return &IncompleteError{Err: err, Missing: missing, Total: total}
}

// batchScratch holds reusable storage for one group of inputs. Each
// concurrent worker owns one scratch value at a time.
type batchScratch struct {
	state      [512]byte
	asmScratch [512]byte
	masks      []uint64
	laneInputs [16][]byte
	tails      [16][128]byte
}

func (b *Batch) getScratch() *batchScratch {
	if v := b.pool.Get(); v != nil {
		return v.(*batchScratch)
	}
	return new(batchScratch)
}

// putScratch returns scratch to the pool. Input references are dropped and the
// staged tails are cleared so neither caller memory nor input bytes are held
// past the end of the call.
func (b *Batch) putScratch(s *batchScratch) {
	for i := range s.laneInputs {
		s.laneInputs[i] = nil
	}
	for i := range s.tails {
		s.tails[i] = [128]byte{}
	}
	b.pool.Put(s)
}

// maskSlice returns a mask slice of length n, reusing scratch storage when
// possible. Existing entries are caller-overwritten, so no stale mask can
// leak between batches.
func (s *batchScratch) maskSlice(n int) []uint64 {
	if cap(s.masks) < n {
		s.masks = make([]uint64, n)
	}
	return s.masks[:n]
}

// batchAVX512 is installed by the amd64 build when the CPU supports AVX512. It
// hashes at most 16 inputs at a time using the shared 16-lane implementation
// and returns the number of digests it fully computed. It is nil everywhere
// else, in which case the portable path transparently takes over.
var batchAVX512 func(ctx context.Context, s *batchScratch, dst [][Size]byte, inputs [][]byte) int

// batchForceGeneric lets tests compare the portable path against the
// hardware path on machines that happen to have AVX512.
var batchForceGeneric bool

// batchAfterDigest, when set, runs after the portable path completes one
// digest. It exists so tests can deterministically interleave cancellation
// with hashing.
var batchAfterDigest func()

// batchGroupObserver, when set, is told when a group's input bytes are
// admitted to the byte budget (acquired=true) and when they are released
// (acquired=false). Tests use it to assert the in-flight byte bound.
var batchGroupObserver func(acquired bool, bytes int)

func (b *Batch) groupSize() int {
	gs := b.opts.Lanes
	if gs > 16 {
		gs = 16
	}
	if b.opts.BatchSize < gs {
		gs = b.opts.BatchSize
	}
	if gs < 1 {
		gs = 1
	}
	return gs
}

// workerCount returns how many lane groups may be in flight at once. It
// rounds down so workers*groupSize never exceeds Lanes: Lanes is a hard bound
// on concurrent digests, not an approximation.
func (b *Batch) workerCount() int {
	gs := b.groupSize()
	workers := b.opts.Lanes / gs
	if workers < 1 {
		workers = 1
	}
	if max := runtime.GOMAXPROCS(0); workers > max {
		workers = max
	}
	return workers
}

// hashGroup computes out for inputs. It returns the number of leading
// digests that were fully computed. Completed digests are written only after
// the whole message has been processed, so cancellation never publishes a
// partial block.
func (b *Batch) hashGroup(ctx context.Context, inputs [][]byte, out [][Size]byte) int {
	if len(inputs) == 0 {
		return 0
	}
	if batchAVX512 != nil && !batchForceGeneric && len(inputs) > 1 {
		// The assembly entry hashes all lanes of the group in lock-step and
		// cannot stop half way through a group. A group is bounded by the
		// lane width, so it finishes promptly; either every digest in the
		// group is published or none is.
		s := b.getScratch()
		got := batchAVX512(ctx, s, out, inputs)
		b.putScratch(s)
		return got
	}

	// The portable path hashes one message at a time and stops between
	// messages. Note it must not use the lock-step hardware routine for a
	// single input either, so tests that force it still get per-input
	// cancellation.
	for i := range inputs {
		if ctx.Err() != nil {
			return i
		}
		out[i] = Sum256(inputs[i])
		if batchAfterDigest != nil {
			batchAfterDigest()
		}
	}
	return len(inputs)
}

// sumContext is the scheduler shared by Sum and Sum256. It returns the
// digests (placeholder entries are zero) and the indices that could not be
// computed.
//
// Small batches (one group, or everything under the byte limit) are hashed
// inline: the SIMD call is only a few microseconds, so spawning goroutines
// would cost more than it saves. Larger batches run through a fixed worker
// pool; the caller stays in the scheduling loop so backpressure admission is
// centralised and cancellation can stop dispatch immediately.
func (b *Batch) sumContext(ctx context.Context, inputs [][]byte) ([][Size]byte, []int) {
	if ctx == nil {
		ctx = context.Background()
	}
	n := len(inputs)
	res := make([][Size]byte, n)
	if n == 0 {
		return res, nil
	}

	groupSize := b.groupSize()
	totalBytes := 0
	for _, in := range inputs {
		totalBytes += len(in)
	}

	// Fast path: a single group fits the lane width and no worker pool or
	// goroutine is needed. A group holding a lone oversized input is hashed
	// inline too; the budget admits it when nothing else is in flight.
	if n <= groupSize && (n == 1 || totalBytes <= b.opts.MaxInputBytes) {
		if ctx.Err() != nil {
			return res, allIndices(n)
		}
		got := b.hashGroup(ctx, inputs, res)
		if got == n {
			return res, nil
		}
		return res, indicesFrom(got, n)
	}

	done := make([]uint32, n)
	budget := newByteBudget(b.opts.MaxInputBytes)
	stopWatch := make(chan struct{})
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		select {
		case <-ctx.Done():
			budget.wakeAll()
		case <-stopWatch:
		}
	}()

	work := make(chan batchJob)
	workers := b.workerCount()
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range work {
				got := b.hashGroup(ctx, job.inputs, res[job.base:job.base+len(job.inputs)])
				for i := 0; i < got; i++ {
					atomic.StoreUint32(&done[job.base+i], 1)
				}
				budget.release(job.bytes)
				if batchGroupObserver != nil {
					batchGroupObserver(false, job.bytes)
				}
			}
		}()
	}

	idx := 0
dispatch:
	for idx < n {
		if ctx.Err() != nil {
			break
		}

		// Build the next group without copying: it is a sub-slice of the
		// caller's input slice. Stop at the byte limit so at most one
		// group's worth of input is admitted at a time beyond the budget.
		start := idx
		groupBytes := 0
		for idx < n && idx-start < groupSize {
			l := len(inputs[idx])
			if idx > start && groupBytes+l > b.opts.MaxInputBytes {
				break
			}
			groupBytes += l
			idx++
		}
		group := inputs[start:idx]

		if !budget.acquire(ctx, groupBytes) {
			idx = start
			break
		}
		if batchGroupObserver != nil {
			batchGroupObserver(true, groupBytes)
		}
		select {
		case work <- batchJob{base: start, inputs: group, bytes: groupBytes}:
		case <-ctx.Done():
			idx = start
			budget.release(groupBytes)
			break dispatch
		}
	}

	close(work)
	wg.Wait()
	close(stopWatch)
	<-watchDone

	var missing []int
	for i := range res {
		if atomic.LoadUint32(&done[i]) == 0 {
			missing = append(missing, i)
		}
	}
	return res, missing
}

// batchJob is one unit of scheduling work: a contiguous group of inputs and
// the byte count admitted to the budget for it.
type batchJob struct {
	base   int
	inputs [][]byte
	bytes  int
}

func allIndices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

func indicesFrom(from, n int) []int {
	idx := make([]int, 0, n-from)
	for i := from; i < n; i++ {
		idx = append(idx, i)
	}
	return idx
}

// byteBudget is a cancellation-aware counting semaphore for input bytes. It
// provides backpressure: admitting a group waits until enough in-flight work
// has drained, instead of buffering the whole input set.
type byteBudget struct {
	mu   sync.Mutex
	cond *sync.Cond
	used int
	max  int
}

func newByteBudget(max int) *byteBudget {
	bb := &byteBudget{max: max}
	bb.cond = sync.NewCond(&bb.mu)
	return bb
}

func (bb *byteBudget) acquire(ctx context.Context, n int) bool {
	bb.mu.Lock()
	defer bb.mu.Unlock()
	for {
		if ctx.Err() != nil {
			return false
		}
		// A single input larger than the configured limit is still allowed
		// to make progress when nothing else is in flight.
		if bb.used == 0 || bb.max <= 0 || bb.used+n <= bb.max {
			bb.used += n
			return true
		}
		bb.cond.Wait()
	}
}

func (bb *byteBudget) release(n int) {
	bb.mu.Lock()
	bb.used -= n
	bb.cond.Broadcast()
	bb.mu.Unlock()
}

// wakeAll unblocks waiters so they can observe ctx.Err. Broadcasting while
// holding the mutex is required: acquire holds the mutex from its ctx check
// through its call to Wait, so a broadcast cannot slip between the check and
// the wait and be lost.
func (bb *byteBudget) wakeAll() {
	bb.mu.Lock()
	bb.cond.Broadcast()
	bb.mu.Unlock()
}
