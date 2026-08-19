package testutil

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRun_Success(t *testing.T) {
	var calls []int
	var resetCalls, stopCalls int
	err := run(2, 3,
		func() { resetCalls++ },
		func() { stopCalls++ },
		func(error) {},
		func(i int) error { calls = append(calls, i); return nil },
	)
	assert.NoError(t, err)
	assert.Equal(t, []int{0, 1, 0, 1, 2}, calls, "warmup indices, then timed indices restart at 0")
	assert.Equal(t, 1, resetCalls)
	assert.Equal(t, 1, stopCalls)
}

func TestRun_ErrorInWarmup(t *testing.T) {
	want := errors.New("boom")
	var stopCalls int
	var fatals []error
	err := run(2, 5,
		func() {},
		func() { stopCalls++ },
		func(e error) { fatals = append(fatals, e) },
		func(i int) error {
			if i == 1 {
				return want
			}
			return nil
		},
	)
	assert.ErrorIs(t, err, want)
	assert.Equal(t, []error{want}, fatals)
	assert.Equal(t, 0, stopCalls, "no timed iterations started, so no stop")
}

func TestRun_ErrorInTimed(t *testing.T) {
	want := errors.New("boom")
	var stopCalls int
	var fatals []error
	err := run(1, 5,
		func() {},
		func() { stopCalls++ },
		func(e error) { fatals = append(fatals, e) },
		func(i int) error {
			if i == 3 {
				return want
			}
			return nil
		},
	)
	assert.ErrorIs(t, err, want)
	assert.Equal(t, []error{want}, fatals)
	assert.Equal(t, 1, stopCalls)
}
