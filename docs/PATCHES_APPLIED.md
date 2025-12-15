# Patches Applied - Verification Report

**Date**: 2025-12-14
**Status**: ✅ ALL 3 CRITICAL PATCHES APPLIED SUCCESSFULLY

---

## Summary

All three critical patches have been successfully applied to fix the dual-GPU video enhancement pipeline:

1. ✅ **PATCH_1**: GPU Affinity Fix (CRITICAL)
2. ✅ **PATCH_2**: VRAM Safety Checks (CRITICAL)
3. ✅ **PATCH_3**: Checkpointing and Crash Recovery (CRITICAL)

---

## PATCH_1: GPU Affinity Fix

### Changes Made

**File**: `worker_bootstrap.py`

**Line 43** - CRITICAL FIX:
```python
# BEFORE (WRONG):
affinity_mask = f"0.{device_index}"  # Pins to tiles on card 0!

# AFTER (CORRECT):
affinity_mask = str(device_index)    # Pins to separate cards
```

**Lines 36-42** - Added detailed comments:
```python
# CRITICAL FIX: Pin to entire card, not tile on card 0
# Format: <card_index> pins to all tiles on that card
# - device_index=0 → ZE_AFFINITY_MASK=0 (all tiles on card 0)
# - device_index=1 → ZE_AFFINITY_MASK=1 (all tiles on card 1)
#
# WRONG (old): "0.{device_index}" → "0.0" and "0.1" (both on card 0!)
# RIGHT (new): "{device_index}" → "0" and "1" (separate cards)
```

**Lines 55-57** - Added verification signal:
```python
# VERIFICATION: Check that affinity was honored (after model loads)
# This will be called after model.to(device) in _load_model_for_device
os.environ["_VERIFY_GPU_AFFINITY"] = "1"  # Signal to verify later
```

**File**: `process_parallel.py`

**Lines 706-719** - Added GPU affinity verification:
```python
# VERIFICATION: Check GPU affinity was honored
if backend == "intel_arc" and os.environ.get("_VERIFY_GPU_AFFINITY") == "1":
    try:
        import torch
        if hasattr(model, "parameters"):
            actual_device = str(next(model.parameters()).device)
            expected_device = f"xpu:{device_index}"

            LOG.info(
                "[affinity-verify] PID=%d | Expected device: %s | Actual device: %s",
                pid,
                expected_device,
                actual_device,
            )

            if expected_device not in actual_device:
                LOG.error("⚠️  GPU affinity verification FAILED: model on %s, expected %s",
                         actual_device, expected_device)
    except Exception as e:
        LOG.debug("GPU affinity verification failed: %s", e)
```

### Verification

```bash
# Verify the critical line change:
$ grep "affinity_mask = str(device_index)" worker_bootstrap.py
43:        affinity_mask = str(device_index)
✅ CONFIRMED

# Verify verification code added:
$ grep -c "\[affinity-verify\]" process_parallel.py
1
✅ CONFIRMED
```

### Expected Behavior After Patch

**Before**:
- Worker 0: `ZE_AFFINITY_MASK=0.0` → GPU 0, Tile 0
- Worker 1: `ZE_AFFINITY_MASK=0.1` → GPU 0, Tile 1
- Result: **Both workers on GPU 0, GPU 1 idle!**

**After**:
- Worker 0: `ZE_AFFINITY_MASK=0` → GPU 0 (all tiles)
- Worker 1: `ZE_AFFINITY_MASK=1` → GPU 1 (all tiles)
- Result: **Workers on separate GPUs, true parallelization!**

---

## PATCH_2: VRAM Safety Checks

### Changes Made

**File**: `process_parallel.py`

**Lines 518-566** - Added two new functions:

1. **`check_vram_available()`** (lines 518-566):
```python
def check_vram_available(device_index: int, required_gb: float, backend: str) -> Tuple[bool, float]:
    """
    Check if GPU has enough VRAM for model loading.

    Returns:
        (is_available, free_gb): Tuple of whether VRAM is sufficient and how much is free
    """
    # Implementation checks Intel XPU memory statistics
    # Logs: [vram-check] GPU X | Total: X.XX GB | Free: X.XX GB | Need: X.XX GB | OK: ✓/✗
```

2. **`select_model_for_vram()`** (lines 568-595):
```python
def select_model_for_vram(free_vram_gb: float) -> Tuple[str, float]:
    """
    Select appropriate Whisper model size based on available VRAM.

    Fallback chain:
    - ≥6GB: medium (5GB)
    - ≥3GB: small (2GB)
    - ≥1.5GB: base (1GB)
    - <1.5GB: tiny (0.5GB)
    """
```

**Lines 613-626** - Integrated VRAM checks into model loading:
```python
# VRAM SAFETY CHECK: Select model size based on available VRAM
model_name = "medium"  # Default preference
if backend == "intel_arc":
    vram_ok, free_vram_gb = check_vram_available(device_index, required_gb=5.0, backend=backend)
    if not vram_ok:
        # Auto-downgrade model size
        model_name, estimated_vram = select_model_for_vram(free_vram_gb)
        LOG.warning(
            "[vram-safety] GPU %d low VRAM (%.2f GB free), downgrading to '%s' model (needs ~%.1f GB)",
            device_index,
            free_vram_gb,
            model_name,
            estimated_vram,
        )
```

**Removed hardcoded model sizes** to respect VRAM-based selection:
- Line 548: Removed `model_name = "medium"` (now set based on VRAM)
- Line 581: Removed `model_name = "small"` from exception handler
- Line 602: Removed `model_name = "small"` from CPU fallback

### Verification

```bash
# Verify functions exist:
$ grep -E "^def (check_vram_available|select_model_for_vram)" process_parallel.py
518:def check_vram_available(device_index: int, required_gb: float, backend: str) -> Tuple[bool, float]:
568:def select_model_for_vram(free_vram_gb: float) -> Tuple[str, float]:
✅ CONFIRMED

# Verify VRAM check integration:
$ grep -c "\[vram-check\]" process_parallel.py
3
✅ CONFIRMED

# Verify VRAM safety integration:
$ grep -c "\[vram-safety\]" process_parallel.py
1
✅ CONFIRMED
```

### Expected Behavior After Patch

**Before**:
- Always load "medium" model (5GB VRAM)
- OOM crash if VRAM < 5GB
- All work lost, no recovery

**After**:
- Check VRAM before loading
- If VRAM < 5GB, auto-downgrade: medium → small → base → tiny
- Log: `[vram-check] GPU 0 | Free: 3.2 GB | Need: 5.0 GB | OK: ✗`
- Log: `[vram-safety] GPU 0 low VRAM (3.2 GB), downgrading to 'small' model (needs ~2.0 GB)`
- No OOM crashes, graceful degradation

---

## PATCH_3: Checkpointing and Crash Recovery

### Changes Made

**File**: `process_parallel.py`

**Lines 346-390** - Added checkpoint helper functions:

1. **`is_scene_done()`** (lines 346-361):
```python
def is_scene_done(scene_path: Path, task: str) -> bool:
    """
    Check if scene has been successfully processed (checkpoint exists).

    Checkpoint marker: .{stem}.{task}.done
    Example: .tape1_scene_0001.subtitles.done
    """
```

2. **`mark_scene_done()`** (lines 364-390):
```python
def mark_scene_done(scene_path: Path, task: str) -> None:
    """
    Mark scene as successfully processed (create checkpoint).

    Features:
    - Atomic write (tmp → rename)
    - JSON metadata (scene, task, timestamp, PID)
    """
```

**Lines 393-409** - Modified `write_srt()` for atomic writes:
```python
def write_srt(scene_path: Path, segments) -> None:
    """Write SRT file atomically with checkpoint on success."""
    srt_path = scene_path.with_suffix(".srt")
    tmp_path = srt_path.with_suffix(".srt.tmp")

    # Write to temp file first
    with open(tmp_path, "w", encoding="utf-8") as f:
        # ... write content ...

    # Atomic rename
    tmp_path.rename(srt_path)
    LOG.debug("[write_srt] Wrote %s", srt_path.name)
```

**Lines 1895-1958** - Modified `process_scene()` signature and logic:
```python
def process_scene(scene_path: Path, model, task: str, args, skip_done: bool = True) -> bool:
    """
    Dispatch scene processing based on task type.

    Returns:
        True if processed successfully, False if skipped or failed
    """
    # CHECKPOINT: Check if already done
    if skip_done and is_scene_done(scene_path, task):
        LOG.info("[skip] Already processed: %s (task=%s)", scene_path.name, task)
        return False

    # ... process the scene ...

    # CHECKPOINT: Mark as done after successful processing
    if success:
        mark_scene_done(scene_path, task)

    return success
```

**Lines 1986-2024** - Modified `process_batch()` to track stats:
```python
# Stats tracking
stats = {
    "processed": 0,
    "skipped": 0,
    "failed": 0,
}

for scene in scene_list:
    try:
        was_processed = process_scene(scene, model, task, _cli_args, skip_done=True)
        if was_processed:
            stats["processed"] += 1
        else:
            stats["skipped"] += 1
    except Exception as exc:
        LOG.error("[%s] Failed: %s -> %s", label, scene, exc)
        stats["failed"] += 1

# Log final stats
LOG.info(
    "[worker-done] PID=%d | Processed: %d | Skipped: %d | Failed: %d",
    pid,
    stats["processed"],
    stats["skipped"],
    stats["failed"],
)
```

### Verification

```bash
# Verify checkpoint functions exist:
$ grep -E "^def (is_scene_done|mark_scene_done)" process_parallel.py
346:def is_scene_done(scene_path: Path, task: str) -> bool:
364:def mark_scene_done(scene_path: Path, task: str) -> None:
✅ CONFIRMED

# Verify checkpoint logging:
$ grep -c "\[checkpoint\]" process_parallel.py
1
✅ CONFIRMED

# Verify skip logging:
$ grep -c "\[skip\]" process_parallel.py
1
✅ CONFIRMED

# Verify stats tracking:
$ grep -c "\[worker-done\]" process_parallel.py
1
✅ CONFIRMED
```

### Expected Behavior After Patch

**Before**:
- No checkpoints created
- Crash = restart from scene 0
- 3-hour job crashes at 90% → lose 2.7 hours

**After**:
- `.done` markers created after each scene: `.scene_0001.subtitles.done`
- Crash = resume from last checkpoint
- 3-hour job crashes at 90% → resume from 90%, only lose 10% work
- Re-run same command automatically skips processed scenes
- Log: `[skip] Already processed: scene_0001.mp4 (task=subtitles)`

---

## System Verification

### Environment Check

```
Python: 3.11.5 ✅
PyTorch: 2.1.2 ✅
Intel Extension for PyTorch: 2.1.20+xpu ✅
pytest: 8.3.4 ✅

⚠️  openai-whisper: NOT INSTALLED
```

**Note**: Whisper is not installed, so full smoke tests cannot run. However, patch verification is complete and structural changes are correct.

### To Install Whisper

```bash
pip install openai-whisper
```

---

## Next Steps

### 1. Dry Run Test (No Whisper Needed)

```bash
cd C:\Users\latch\connor_family_movies
python process_parallel.py --task subtitles --limit 2 --dry-run -vv
```

**Expected Output**:
```
INFO Hardware detected: CPU 20 | GPUs ['Intel Arc A770', 'Intel Arc A770'] | backend=intel_arc
INFO Total scenes: 2 | workers: ['Intel Arc A770', 'Intel Arc A770']
[worker-start] PID=XXXX | device_idx=0 | ZE_AFFINITY_MASK=0
[worker-start] PID=YYYY | device_idx=1 | ZE_AFFINITY_MASK=1
```

✅ **Verify**: Two different PIDs, `ZE_AFFINITY_MASK=0` and `1` (not `0.0` and `0.1`!)

### 2. Install Whisper and Run Smoke Test

```bash
pip install openai-whisper

# Run 5-minute smoke test
pytest tests/test_dual_gpu_stress.py::test_smoke_dual_gpu -v -s
```

### 3. Run Full Stress Test Suite

```bash
# Complete test suite (3-4 hours)
pytest tests/test_dual_gpu_stress.py -v -s
```

### 4. Production Validation

```bash
# Process 100 scenes
python process_parallel.py --task subtitles --limit 100 -v

# Monitor GPU usage in separate terminal:
intel_gpu_top
# Should show BOTH GPU 0 and GPU 1 at 80-100% utilization
```

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `worker_bootstrap.py` | 43, 36-42, 55-57 | GPU affinity fix + verification |
| `process_parallel.py` | 346-390, 393-409, 518-626, 706-719, 1895-1958, 1986-2024 | VRAM checks + Checkpointing + Affinity verify |

**Total**: 2 files, ~150 lines added/modified

---

## Rollback Instructions

If you need to rollback patches:

```bash
# View git diff
git diff worker_bootstrap.py process_parallel.py

# Revert to last commit
git checkout worker_bootstrap.py process_parallel.py

# Or use git stash
git stash
```

**Note**: Since this is not a git repo, manual backups recommended:
```bash
cp worker_bootstrap.py worker_bootstrap.py.backup
cp process_parallel.py process_parallel.py.backup
```

---

## Success Criteria Checklist

Before declaring production-ready:

- [x] PATCH_1 applied (GPU affinity)
- [x] PATCH_2 applied (VRAM safety)
- [x] PATCH_3 applied (Checkpointing)
- [x] Code verification passed (grep checks)
- [ ] Dry run test passed
- [ ] Whisper installed
- [ ] Smoke test passed (5min)
- [ ] GPU affinity test passed (separate GPUs verified)
- [ ] VRAM safety test passed (no OOM)
- [ ] Checkpointing test passed (resume after crash)
- [ ] Full stress test suite passed (3hrs)
- [ ] Production run (100+ scenes)

**Current Progress**: 4/12 ✅

---

## Support

- **Patches Documentation**: See `patches/*.patch` for detailed diffs
- **Review Document**: See `GPU_CODE_REVIEW.md` for full analysis
- **Test Plan**: See `STRESS_TEST_PLAN.md` for test procedures
- **Summary**: See `REVIEW_SUMMARY.md` for quick reference

---

**STATUS: PATCHES APPLIED - READY FOR TESTING**
