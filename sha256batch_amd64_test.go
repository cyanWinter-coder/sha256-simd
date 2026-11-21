//go:build !noasm && !appengine && gc
// +build !noasm,!appengine,gc

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
	"math/rand"
	"testing"
)

// TestGenMaskIntoMatchesGenMask pins the allocation-free mask generator to the
// original one, which the single-stream AVX512 path already uses.
func TestGenMaskIntoMatchesGenMask(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for iter := 0; iter < 500; iter++ {
		var inputs [16][]byte
		for i := range inputs {
			// Zero block counts and mismatched block counts are both legal.
			inputs[i] = make([]byte, r.Intn(8)*BlockSize)
		}
		got := expandMaskInto(new(batchScratch), genMaskInto(inputs))
		want := expandMask(genMask(inputs))
		if len(got) != len(want) {
			t.Fatalf("iter %d: mask lengths differ: got %d want %d", iter, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("iter %d index %d: got %016x want %016x", iter, i, got[i], want[i])
			}
		}
	}
}

// TestBatchAVX512MixedLanes feeds the hardware path lanes that differ in block
// count, empty lanes, and a full 16-lane group, and checks every digest.
func TestBatchAVX512MixedLanes(t *testing.T) {
	if batchAVX512 == nil {
		t.Skip("AVX512 not available on this CPU")
	}
	b, err := NewBatch(BatchOptions{Lanes: 16, BatchSize: 16, MaxInputBytes: 1 << 24})
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(99))
	for iter := 0; iter < 100; iter++ {
		n := 1 + r.Intn(16)
		inputs := make([][]byte, n)
		for i := range inputs {
			var l int
			switch i % 4 {
			case 0:
				l = 0
			case 1:
				l = r.Intn(BlockSize)
			case 2:
				l = r.Intn(3 * BlockSize)
			default:
				l = batchWindowBytes + r.Intn(2*BlockSize) + BlockSize
			}
			inputs[i] = make([]byte, l)
			r.Read(inputs[i])
		}
		got, err := b.Sum256(context.Background(), inputs)
		if err != nil {
			t.Fatalf("iter %d: %v", iter, err)
		}
		for i := range inputs {
			if want := sha256.Sum256(inputs[i]); got[i] != want {
				t.Fatalf("iter %d input %d (len %d): got %x want %x", iter, i, len(inputs[i]), got[i], want)
			}
		}
	}
}

// TestBatchAVX512ScratchIsolation reuses one pooled scratch value across
// groups with completely different lane counts and content. A stale lane or
// tail must not leak into the next group.
func TestBatchAVX512ScratchIsolation(t *testing.T) {
	if batchAVX512 == nil {
		t.Skip("AVX512 not available on this CPU")
	}
	s := new(batchScratch)
	// Warm the scratch with a full, long group.
	warm := make([][]byte, 16)
	for i := range warm {
		warm[i] = make([]byte, 4*BlockSize)
		for j := range warm[i] {
			warm[i][j] = 0xAA
		}
	}
	out := make([][Size]byte, 16)
	if n := batchAVX512(context.Background(), s, out, warm); n != 16 {
		t.Fatalf("warm-up returned %d", n)
	}

	for _, n := range []int{1, 2, 5, 16} {
		inputs := make([][]byte, n)
		for i := range inputs {
			inputs[i] = make([]byte, (i%3)*BlockSize+i)
			for j := range inputs[i] {
				inputs[i][j] = byte(i + j)
			}
		}
		got := make([][Size]byte, n)
		if gotN := batchAVX512(context.Background(), s, got, inputs); gotN != n {
			t.Fatalf("n=%d: returned %d", n, gotN)
		}
		for i := range inputs {
			if want := sha256.Sum256(inputs[i]); got[i] != want {
				t.Fatalf("n=%d index %d: got %x want %x", n, i, got[i], want)
			}
		}
	}
}

// TestBatchAllImplementationsAgree hashes the same inputs with the AVX512
// backend, the portable backend on top of the Intel SHA extension, and the
// pure-Go block function. All three must be byte-for-byte identical.
func TestBatchAllImplementationsAgree(t *testing.T) {
	inputs := randomInputs(1234, 40, 3*BlockSize)
	// Cover padding boundaries and one multi-window message.
	for _, l := range []int{0, 1, 55, 56, 63, 64, 65, 119, 120, 128, batchWindowBytes + 1} {
		msg := make([]byte, l)
		rand.New(rand.NewSource(int64(l))).Read(msg)
		inputs = append(inputs, msg)
	}

	want := make([][Size]byte, len(inputs))
	for i := range inputs {
		want[i] = sha256.Sum256(inputs[i])
	}

	run := func(name string, generic bool, block blockfuncType) {
		oldForce, oldBlock := batchForceGeneric, blockfunc
		batchForceGeneric = generic
		blockfunc = block
		defer func() {
			batchForceGeneric, blockfunc = oldForce, oldBlock
		}()
		// Re-normalize after restoring in case a test changed it.
		b, err := NewBatch(DefaultBatchOptions())
		if err != nil {
			t.Fatal(err)
		}
		got, err := b.Sum256(context.Background(), inputs)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: index %d (len %d): got %x want %x", name, i, len(inputs[i]), got[i], want[i])
			}
		}
	}

	if batchAVX512 == nil {
		t.Skip("AVX512 not available on this CPU")
	}
	run("avx512", false, blockfuncIntelSha)
	run("generic", true, blockfuncForceGeneric)
	if hasIntelSha {
		run("intel-sha", true, blockfuncIntelSha)
	}
}
