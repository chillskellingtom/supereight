# Phase 1: GPU Affinity Fix - COMPLETE

## Summary

Successfully implemented multiprocessing-based GPU affinity for dual Intel Arc A770 GPUs.

## What Was Fixed

### Critical Bug: Thread-Based GPU Affinity
**Problem**: The original implementation used `threading.Thread`, which cannot maintain independent GPU affinity because:
- All threads share the same process memory space
- `ZE_AFFINITY_MASK` is a process-level environment variable
- When one thread sets `ZE_AFFINITY_MASK=0.0` and another sets `ZE_AFFINITY_MASK=0.1`, they overwrite each other
- Additionally, `ZE_AFFINITY_MASK` was set AFTER importing torch, which is too late (GPU runtime initialization happens at import time)

**Evidence**: See `tests/test_gpu_affinity.py::test_threads_share_env_variables`
```
Worker 1: intended='0.0', actual='0.1'  ← Bug: Worker 2 overwrote Worker 1's setting
Worker 2: intended='0.1', actual='0.1'
Same PID: 19924  ← Both threads in same process
```

### Solution: Process-Based GPU Affinity
**Implementation**: Replaced `threading.Thread` with `multiprocessing.Process`

**Proof**: See `tests/test_multiprocessing_affinity.py::test_processes_have_independent_env_variables`
```
Worker 1 PID 4384: intended='0.0', actual='0.0' ✓
Worker 2 PID 26656: intended='0.1', actual='0.1' ✓
Different PIDs ← Separate processes, independent environments
```

## Changes Made

### 1. Added `--limit` Flag (TDD)
**File**: `process_parallel.py`
- Parser accepts `--limit N` to process only first N scenes
- `gather_scenes()` respects limit
- **Tests**: `tests/test_smoke_subtitles.py::TestLimit` (all passing)

### 2. Created Worker Bootstrap Module
**File**: `worker_bootstrap.py`
- Sets `ZE_AFFINITY_MASK` BEFORE any GPU imports
- Loads process_parallel module AFTER environment is configured
- Each worker process gets isolated environment

**Key Design**:
```python
def worker_main(device, scene_list, backend, task, cli_args):
    # 1. Set affinity BEFORE imports
    set_gpu_affinity(device, backend)  # Sets ZE_AFFINITY_MASK here

    # 2. NOW import GPU libraries (torch, ipex, etc.)
    import process_parallel

    # 3. Process scenes with correct GPU affinity
    process_parallel.process_batch(...)
```

### 3. Refactored main() to Use Multiprocessing
**File**: `process_parallel.py::main()`
- Removed `threading.Thread`
- Added `multiprocessing.get_context("spawn")` for Windows compatibility
- Workers are now `multiprocessing.Process` instances
- Each worker runs `worker_bootstrap.worker_main()`

### 4. Added Worker Identity Logging
**File**: `process_parallel.py::process_batch()`
- Logs PID, thread/process name, device index, ZE_AFFINITY_MASK
- Helps debug GPU affinity issues
- Example log:
  ```
  [worker-start] PID=4384 | thread=Process-1 | backend=intel_arc | device_idx=0 (Intel Arc A770) | ZE_AFFINITY_MASK=0.0
  ```

### 5. Created Test Infrastructure
**Files**:
- `tests/test_smoke_subtitles.py` - Basic smoke tests
- `tests/test_gpu_affinity.py` - Demonstrates threading bug
- `tests/test_multiprocessing_affinity.py` - Proves multiprocessing fix
- `scripts/smoke_subtitles_tape1_1to20.ps1` - Smoke test script for tape 1 scenes 1-20

## Validation

### Dry Run Successful
```bash
$ python process_parallel.py --task subtitles --limit 2 --dry-run -v

21:24:00 INFO Hardware detected: CPU 20 | GPUs ['Intel(R) Arc(TM) A770 Graphics', 'Intel(R) Arc(TM) A770 Graphics'] | backend=intel_arc
21:24:00 INFO Intel Arc GPUs detected: 2 device(s) - Intel(R) Arc(TM) A770 Graphics (ze_index=0), Intel(R) Arc(TM) A770 Graphics (ze_index=1)
21:24:00 INFO Total scenes: 2 | workers: ['Intel(R) Arc(TM) A770 Graphics', 'Intel(R) Arc(TM) A770 Graphics']
```

### Test Results
All tests passing:
- ✓ `test_limit_flag_accepted_by_parser`
- ✓ `test_limit_defaults_to_none`
- ✓ `test_threads_share_env_variables` (demonstrates bug)
- ✓ `test_processes_have_independent_env_variables` (proves fix)

## Known Limitations

### Monitor Thread Disabled
The monitor thread (progress/ETA display) has been temporarily disabled because it relied on shared global variables (`_progress_lock`, `_processed`) that don't work across processes.

**TODO for Phase 2**: Implement monitor using multiprocessing.Queue for inter-process communication

### No Live Testing with Whisper Yet
The smoke test requires `openai-whisper` which is not installed. The multiprocessing refactor has been validated with:
- Dry runs showing correct device detection
- Unit tests proving process isolation
- Structural code review

**TODO for Phase 2**: Install whisper and run full smoke test

## Next Steps - Phase 2

1. **Restore Progress Monitoring**
   - Replace shared globals with `multiprocessing.Queue`
   - Each worker sends progress updates to queue
   - Monitor process reads from queue and displays stats

2. **Install Whisper and Run Full Smoke Test**
   ```bash
   pip install openai-whisper
   scripts/smoke_subtitles_tape1_1to20.ps1
   ```
   - Validate GPU affinity in real processing
   - Check logs for `[worker-start]` entries showing different PIDs
   - Verify both GPUs are utilized (check `ZE_AFFINITY_MASK=0.0` and `ZE_AFFINITY_MASK=0.1`)

3. **Add OOM/VRAM Resilience** (Phase 3 from original plan)
   - Implement fallback chains for model sizing
   - Add VRAM monitoring where available
   - Graceful degradation on OOM

4. **Add Checkpointing** (Phase 4)
   - Per-worker checkpoint files
   - Atomic writes for crash recovery
   - Skip already-processed scenes on restart

5. **Implement Dynamic Load Balancing** (Phase 5)
   - Replace round-robin with work-stealing queue
   - Use `multiprocessing.JoinableQueue`
   - Workers pull from shared queue instead of pre-assigned batches

## Files Modified

- `process_parallel.py` - Main refactor (threading → multiprocessing)
- `worker_bootstrap.py` - NEW: Worker process entrypoint
- `tests/test_smoke_subtitles.py` - NEW: Limit flag tests
- `tests/test_gpu_affinity.py` - NEW: Demonstrates threading bug
- `tests/test_multiprocessing_affinity.py` - NEW: Proves multiprocessing fix
- `scripts/smoke_subtitles_tape1_1to20.ps1` - NEW: Smoke test launcher

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│ Main Process (process_parallel.py)                 │
│ - Detects hardware (2x Arc A770)                   │
│ - Builds worker plan (round-robin scenes)          │
│ - Spawns worker processes                          │
└──────────────┬──────────────────────────────────────┘
               │
               ├─────────────────────────┬─────────────────────────┐
               │                         │                         │
               ▼                         ▼                         ▼
    ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │ Worker Process 1 │     │ Worker Process 2 │     │ (Future: Monitor)│
    │ PID: 4384        │     │ PID: 26656       │     │                  │
    ├──────────────────┤     ├──────────────────┤     └──────────────────┘
    │ worker_bootstrap │     │ worker_bootstrap │
    │ ┌──────────────┐ │     │ ┌──────────────┐ │
    │ │1. Set env    │ │     │ │1. Set env    │ │
    │ │  ZE_AFFINITY │ │     │ │  ZE_AFFINITY │ │
    │ │  _MASK=0.0   │ │     │ │  _MASK=0.1   │ │
    │ └──────────────┘ │     │ └──────────────┘ │
    │ ┌──────────────┐ │     │ ┌──────────────┐ │
    │ │2. Import GPU │ │     │ │2. Import GPU │ │
    │ │  libs (torch)│ │     │ │  libs (torch)│ │
    │ └──────────────┘ │     │ └──────────────┘ │
    │ ┌──────────────┐ │     │ ┌──────────────┐ │
    │ │3. Load model │ │     │ │3. Load model │ │
    │ │  on GPU 0    │ │     │ │  on GPU 1    │ │
    │ └──────────────┘ │     │ └──────────────┘ │
    │ ┌──────────────┐ │     │ ┌──────────────┐ │
    │ │4. Process    │ │     │ │4. Process    │ │
    │ │  scenes 0,2..│ │     │ │  scenes 1,3..│ │
    │ └──────────────┘ │     │ └──────────────┘ │
    └──────────────────┘     └──────────────────┘
         GPU 0                     GPU 1
```

## How to Verify the Fix

1. **Check Different PIDs**:
   ```bash
   python process_parallel.py --task subtitles --limit 4 -vv 2>&1 | grep "worker-start"
   ```
   Should show different PIDs:
   ```
   [worker-start] PID=4384 | ... | device_idx=0 | ZE_AFFINITY_MASK=0.0
   [worker-start] PID=5678 | ... | device_idx=1 | ZE_AFFINITY_MASK=0.1
   ```

2. **Check GPU Utilization** (requires actual processing):
   ```bash
   # In one terminal:
   scripts/smoke_subtitles_tape1_1to20.ps1

   # In another terminal:
   nvidia-smi  # Or equivalent for Intel Arc
   ```
   Both GPUs should show activity during processing.

3. **Run All Tests**:
   ```bash
   pytest tests/ -v
   ```
   All tests should pass.

## Conclusion

Phase 1 successfully addressed the **structural impossibility** of GPU affinity with threading by:
- Proving the bug with tests (`test_gpu_affinity.py`)
- Implementing the fix with multiprocessing (`worker_bootstrap.py`, refactored `main()`)
- Validating the fix with tests (`test_multiprocessing_affinity.py`)
- Dry-run confirming correct device detection and worker assignment

The foundation is now solid for Phase 2 (progress monitoring, full smoke test) and beyond.
