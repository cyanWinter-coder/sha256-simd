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
	"encoding/binary"
)

func init() {
	if hasAvx512 {
		batchAVX512 = batchAVX512Impl
	}
}

var avx512Init = [8]uint32{
	init0, init1, init2, init3, init4, init5, init6, init7,
}

// runAVX512 hashes the staged lane inputs. Their lengths must be multiples of
// BlockSize; the assembly reads them in place.
func runAVX512(s *batchScratch) {
	mr := genMaskInto(s.laneInputs)
	mask := expandMaskInto(s, mr)
	if len(mask) == 0 {
		return
	}
	sha256X16Avx512(&s.state, &s.asmScratch, &table, mask, s.laneInputs)
}

// genMaskInto produces the same mask sequence as genMask without the sorting
// interface (and its per-call allocation). Lanes are sorted ascending by
// length with a 16-element insertion sort; zero-length lanes are cleared
// before any mask is emitted and therefore never appear in one.
func genMaskInto(input [16][]byte) (mr [16]maskRounds) {
	var sorted [16]lane
	k := 0
	var m uint64
	for c, in := range input {
		l := len(in)
		if l == 0 {
			continue
		}
		pos := k
		for pos > 0 && sorted[pos-1].len > uint(l) {
			sorted[pos] = sorted[pos-1]
			pos--
		}
		sorted[pos] = lane{uint(l), uint(c)}
		k++
		m |= 1 << uint(c)
	}

	round, index := uint64(0), 0
	for _, ln := range sorted[:k] {
		if blocks := uint64(ln.len) >> 6; blocks > round {
			mr[index] = maskRounds{m, blocks - round}
			index++
			round = blocks
		}
		m &^= 1 << ln.pos
	}
	return
}

// expandMaskInto is expandMask without the allocation.
func expandMaskInto(s *batchScratch, mr [16]maskRounds) []uint64 {
	size := 0
	for _, r := range mr {
		size += int(r.rounds)
	}
	if size == 0 {
		return nil
	}
	mask := s.maskSlice(size)
	index := 0
	for _, r := range mr {
		for j := uint64(0); j < r.rounds; j++ {
			mask[index] = r.mask
			index++
		}
	}
	return mask
}

// batchAVX512Impl hashes up to 16 inputs using the 16-way assembly routine.
//
// Complete blocks are streamed straight from the caller's slices, so the bulk
// of the input is never copied. Only the padding tail of each message (at most
// two blocks) is staged in scratch memory. The lane state and scratch buffers
// are fully overwritten on every call, so a pooled scratch value cannot leak
// data from a previous batch.
//
// It returns the number of inputs whose digest was fully computed and written
// to dst. Lanes are processed in lock-step, so on cancellation the group
// either finishes or none of its digests are published; a partially hashed
// message is never returned.
func batchAVX512Impl(ctx context.Context, s *batchScratch, dst [][Size]byte, inputs [][]byte) int {
	n := len(inputs)
	for i := 0; i < 16; i++ {
		s.laneInputs[i] = nil
	}
	if n == 0 {
		return 0
	}

	// Reset every lane to the SHA-256 initial state, little endian.
	for k := 0; k < 8; k++ {
		for i := 0; i < 16; i++ {
			binary.LittleEndian.PutUint32(s.state[(i+k*16)*4:], avx512Init[k])
		}
	}

	var off [16]int
	var total [16]uint64
	for i := 0; i < n; i++ {
		total[i] = uint64(len(inputs[i]))
	}

	// Hash complete blocks through the assembly routine, window by window.
	for {
		window := 0
		for i := 0; i < n; i++ {
			full := (int(total[i]) - off[i]) / BlockSize * BlockSize
			if full > batchWindowBytes {
				full = batchWindowBytes
			}
			if full > window {
				window = full
			}
			s.laneInputs[i] = inputs[i][off[i] : off[i]+full]
		}
		if window == 0 {
			break
		}
		select {
		case <-ctx.Done():
			for i := 0; i < 16; i++ {
				s.laneInputs[i] = nil
			}
			return 0
		default:
		}
		runAVX512(s)
		for i := 0; i < n; i++ {
			off[i] += len(s.laneInputs[i])
			s.laneInputs[i] = nil
		}
	}

	// Feed the final padded block(s) of each message. Padding is staged in
	// scratch memory; it is at most 128 bytes per lane.
	for i := 0; i < n; i++ {
		rem := int(total[i]) - off[i]
		tail := s.tails[i][:]
		for j := range tail {
			tail[j] = 0
		}
		copy(tail, inputs[i][off[i]:])
		tail[rem] = 0x80

		tailLen := BlockSize
		if rem >= BlockSize-8 {
			tailLen = 2 * BlockSize
		}
		binary.BigEndian.PutUint64(tail[tailLen-8:], total[i]<<3)
		s.laneInputs[i] = tail[:tailLen]
	}
	runAVX512(s)
	for i := 0; i < 16; i++ {
		s.laneInputs[i] = nil
	}

	for i := 0; i < n; i++ {
		dst[i] = getDigest(i, s.state[:])
	}
	return n
}
