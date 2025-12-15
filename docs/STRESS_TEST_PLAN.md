# 2× GPU Stress Test Plan

## Overview
Comprehensive test plan for validating dual Intel Arc A770 parallel video enhancement pipeline.

**Hardware**: Gigabyte Z790 AORUS, i5-13600K (14C/20T), 2× Intel Arc A770 (16GB each)

---

## Pre-Flight Checklist

### 1. Apply Critical Patches
```bash
# Apply patches in order
cd C:\Users\latch\connor_family_movies\patches

# Patch 1: Fix GPU affinity (CRITICAL)
git apply PATCH_1_fix_gpu_affinity.patch

# Patch 2: Add VRAM safety checks
git apply PATCH_2_vram_safety_check.patch

# Patch 3: Implement checkpointing
git apply PATCH_3_checkpointing.patch
```

### 2. Verify Environment
```bash
# Check Python version
python --version  # Should be 3.8+

# Check packages
pip list | grep -E "torch|intel-extension|whisper"

# Check GPUs detected
python -c "import intel_extension_for_pytorch as ipex; print(f'GPUs: {ipex.xpu.device_count()}')"

# Should output: GPUs: 2
```

### 3. Prepare Test Data
```bash
# Ensure scenes folder exists with test videos
ls -la "C:\Users\latch\connor_family_movies_processed\scenes" | head -20

# Minimum: 20 scenes for load balancing tests
# Recommended: 100 scenes for memory leak tests
```

---

## Test Suite Execution

### Quick Smoke Test (5 minutes)
**Purpose**: Fast sanity check before full test suite

```bash
pytest tests/test_dual_gpu_stress.py::test_smoke_dual_gpu -v -s
```

**Expected Output**:
```
SMOKE TEST: Dual-GPU Basic Functionality
================================================================================
[worker-start] PID=4384 | device_idx=0 | ZE_AFFINITY_MASK=0
[worker-start] PID=5678 | device_idx=1 | ZE_AFFINITY_MASK=1
[model-load] PID=4384 | device: xpu:0
[model-load] PID=5678 | device: xpu:1

✅ SMOKE TEST PASSED - Basic dual-GPU functionality working
```

**Success Criteria**:
- [x] 2 workers with different PIDs
- [x] Models loaded on xpu:0 and xpu:1
- [x] No crashes

---

### Test 1: GPU Affinity Verification (10 minutes)
**Purpose**: Verify workers use separate GPUs, not same GPU with different tiles

```bash
pytest tests/test_dual_gpu_stress.py::test_gpu_affinity_dual_workers -v -s
```

**What This Tests**:
1. Two worker processes spawn correctly
2. Each worker gets unique PID
3. Worker 0: `ZE_AFFINITY_MASK=0`, model on `xpu:0`
4. Worker 1: `ZE_AFFINITY_MASK=1`, model on `xpu:1`

**Bug Detection**:
- ❌ **FAIL**: Both workers `ZE_AFFINITY_MASK=0.0` and `0.1` (same GPU, different tiles)
- ❌ **FAIL**: Both models on `xpu:0` (affinity not honored)
- ✅ **PASS**: Workers on `xpu:0` and `xpu:1` (separate GPUs)

**Expected Output**:
```
TEST 1: GPU AFFINITY VERIFICATION
================================================================================
📊 Found 2 workers
  Worker PID=4384: device_idx=0, ZE_AFFINITY_MASK=0
  Worker PID=5678: device_idx=1, ZE_AFFINITY_MASK=1

📦 Found 2 model loads
  PID=4384: model=medium, device=xpu:0
  PID=5678: model=medium, device=xpu:1

✅ Found 2 affinity verifications
  ✓ PID=4384: expected=xpu:0, actual=xpu:0
  ✓ PID=5678: expected=xpu:1, actual=xpu:1

✅ GPU AFFINITY TEST PASSED
   - 2 workers with different PIDs
   - Workers use device 0 and 1
   - ZE_AFFINITY_MASK correctly set (0 and 1)
   - Models loaded on xpu:0 and xpu:1
```

**Metrics to Monitor**:
```bash
# In separate terminal, monitor GPU usage
intel_gpu_top

# Should show:
# GPU 0: 80-100% busy (Worker 0)
# GPU 1: 80-100% busy (Worker 1)
```

---

### Test 2: VRAM Utilization (15 minutes)
**Purpose**: Verify VRAM safety checks prevent OOM crashes

```bash
pytest tests/test_dual_gpu_stress.py::test_vram_utilization -v -s
```

**What This Tests**:
1. VRAM checks before model loading
2. Automatic model downgrade on low VRAM
3. No OOM crashes

**Expected Output**:
```
TEST 2: VRAM UTILIZATION
================================================================================
📊 Found 2 VRAM checks
  GPU 0: 12.50 GB free, need 5.00 GB - ✓
  GPU 1: 12.50 GB free, need 5.00 GB - ✓

✅ VRAM UTILIZATION TEST PASSED
   - No OOM crashes
   - 2 VRAM checks performed
```

**Simulated Low VRAM Test** (manual):
```python
# Temporarily fill VRAM to test downgrade
import torch
import intel_extension_for_pytorch as ipex

# Fill VRAM with dummy tensors
dummy_tensors = []
for i in range(10):
    t = torch.randn(1000, 1000, 1000, device='xpu:0')  # ~4GB
    dummy_tensors.append(t)

# Now run test - should downgrade to small model
# Expected: [vram-safety] downgrading to 'small' model
```

---

### Test 3: Load Balancing (30 minutes)
**Purpose**: Verify work distributed evenly across GPUs

```bash
pytest tests/test_dual_gpu_stress.py::test_load_balancing -v -s
```

**What This Tests**:
1. Both workers process approximately equal scenes
2. Parallel speedup (2× throughput vs single GPU)
3. No idle workers

**Expected Output**:
```
TEST 3: LOAD BALANCING AND THROUGHPUT
================================================================================
📊 Load Distribution:
  Worker PID=4384: 10 scenes
  Worker PID=5678: 10 scenes

⚖️  Load Balance Ratio: 1.00 (1.0 = perfect balance)

⏱️  Total Time: 180.5s (3.0min)
   Average per scene: 9.0s

✅ LOAD BALANCING TEST PASSED
```

**Metric**: Load Balance Ratio
- **Perfect**: 1.0 (both workers equal scenes)
- **Good**: ≥ 0.9 (within 10% difference)
- **Poor**: < 0.8 (significant imbalance)

**Baseline Comparison**:
```bash
# Single-GPU baseline (for speedup calculation)
python process_parallel.py --task subtitles --limit 20 -v 2>&1 | grep "elapsed"

# Expected:
# Single GPU: ~360s (6min) for 20 scenes
# Dual GPU: ~180s (3min) for 20 scenes
# Speedup: 2.0× (ideal)
```

---

### Test 4: Checkpointing and Resume (20 minutes)
**Purpose**: Verify crash recovery via checkpointing

```bash
pytest tests/test_dual_gpu_stress.py::test_checkpointing_and_resume -v -s
```

**What This Tests**:
1. `.done` markers created after each scene
2. Scenes skipped on re-run (not re-processed)
3. Atomic writes (no corruption on crash)

**Expected Output**:
```
TEST 4: CHECKPOINTING AND CRASH RECOVERY
================================================================================
▶️  First run: processing 10 scenes...

📍 Created 10 checkpoint markers

▶️  Second run: should skip all scenes...

⏭️  Skipped 10 scenes

✅ CHECKPOINTING TEST PASSED
   - 10 checkpoint markers created
   - All scenes skipped on re-run
```

**Manual Crash Test**:
```bash
# Start processing 20 scenes
python process_parallel.py --task subtitles --limit 20 -v &
PID=$!

# Wait for 5 scenes to complete
sleep 60

# Simulate crash (kill process)
kill -9 $PID

# Count checkpoints
ls -1 scenes/.*.subtitles.done | wc -l
# Should show ~5 .done markers

# Resume (should skip first 5, process remaining 15)
python process_parallel.py --task subtitles --limit 20 -v

# Check logs for: "[skip] Already processed: scene_000X.mp4"
```

---

### Test 5: Memory Leak Detection (60+ minutes)
**Purpose**: Detect memory leaks during long-running processing

```bash
pytest tests/test_dual_gpu_stress.py::test_memory_leak_detection -v -s
```

**What This Tests**:
1. Memory usage over 100-scene batch
2. No unbounded growth (leak detection)
3. Proper cleanup on worker exit

**Expected Output**:
```
TEST 5: MEMORY LEAK DETECTION
================================================================================
⏳ Running 100-scene batch with memory monitoring...
   (This may take 30-60 minutes)

  [0s] Memory: 2500.0 MB (3 processes)
  [30s] Memory: 2800.0 MB (3 processes)
  [60s] Memory: 2850.0 MB (3 processes)
  ...
  [1800s] Memory: 3200.0 MB (3 processes)

📈 Memory Analysis:
   Initial: 2500.0 MB
   Final: 3200.0 MB
   Peak: 3400.0 MB
   Growth: 700.0 MB (+28.0%)

✅ MEMORY LEAK TEST PASSED
```

**Leak Detection Criteria**:
- **PASS**: Growth < 50% (acceptable for caching/buffers)
- **WARN**: Growth 50-100% (possible slow leak, investigate)
- **FAIL**: Growth > 100% (definite leak, must fix)

**Tools for Manual Monitoring**:
```bash
# Monitor system memory
watch -n 5 free -h

# Monitor process memory
watch -n 5 'ps aux | grep process_parallel | grep -v grep'

# Monitor GPU memory
watch -n 5 intel_gpu_top
```

---

### Test 6: Determinism and Reproducibility (20 minutes)
**Purpose**: Verify identical results across runs

```bash
# Run 1
python process_parallel.py --task subtitles --limit 5 -v --seed 42

# Run 2 (same seed, should produce identical outputs)
python process_parallel.py --task subtitles --limit 5 -v --seed 42

# Compare outputs
diff -r scenes/tape1_scene_0001.srt scenes/tape1_scene_0001.srt.run2
# Should output: (no differences)
```

**Success Criteria**:
- [x] Identical SRT files across runs (same seed)
- [x] Identical transcripts (word-for-word)
- [x] Identical timestamps (frame-accurate)

**Note**: This test requires `--seed` flag implementation (from Issue #9).

---

## Continuous Monitoring

### GPU Utilization Dashboard
```bash
# Terminal 1: Run processing
python process_parallel.py --task subtitles --limit 100 -v

# Terminal 2: Monitor GPUs
watch -n 1 intel_gpu_top

# Terminal 3: Monitor system
watch -n 1 'ps aux | grep -E "(process_parallel|python)" | grep -v grep'
```

### Log Analysis
```bash
# Extract worker assignments
grep "\[worker-start\]" output.log

# Extract processing times
grep "\[progress\]" output.log | awk '{print $NF}'

# Extract errors
grep -i "error\|exception\|failed" output.log
```

---

## Success Criteria Summary

| Test | Duration | Critical? | Pass Criteria |
|------|----------|-----------|---------------|
| Smoke Test | 5min | YES | No crashes, 2 workers, 2 GPUs |
| GPU Affinity | 10min | YES | ZE_AFFINITY_MASK=0,1; models on xpu:0,xpu:1 |
| VRAM Safety | 15min | YES | No OOM; auto-downgrade working |
| Load Balancing | 30min | NO | Balance ratio ≥ 0.8; speedup ≥ 1.8× |
| Checkpointing | 20min | YES | .done markers created; scenes skipped |
| Memory Leaks | 60min | YES | Memory growth < 50% |
| Determinism | 20min | NO | Identical outputs (same seed) |

**Overall**: 6/7 tests must PASS for production readiness. Load balancing and determinism are nice-to-have optimizations.

---

## Troubleshooting Guide

### Issue: Both Workers Use GPU 0
**Symptoms**: Both models on `xpu:0`, GPU 1 idle
**Diagnosis**: Patch #1 not applied correctly
**Fix**:
```bash
# Verify worker_bootstrap.py line 44:
grep "affinity_mask = str(device_index)" worker_bootstrap.py

# Should output:
#   affinity_mask = str(device_index)  # NOT f"0.{device_index}"
```

### Issue: OOM Crashes
**Symptoms**: `RuntimeError: out of memory`
**Diagnosis**: VRAM checks not working (patch #2 missing)
**Fix**:
```bash
# Verify VRAM check function exists:
grep "def check_vram_available" process_parallel.py

# Manual workaround: use small model
python process_parallel.py --task subtitles --model small
```

### Issue: Scenes Re-Processed After Crash
**Symptoms**: No `.done` markers, all scenes re-processed
**Diagnosis**: Checkpoint code not active (patch #3 missing)
**Fix**:
```bash
# Verify checkpoint functions exist:
grep -E "def (is_scene_done|mark_scene_done)" process_parallel.py

# Manual workaround: track manually
touch scenes/.scene_0001.subtitles.done
```

### Issue: Load Imbalance (ratio < 0.8)
**Symptoms**: Worker 1 finishes early, Worker 2 still processing
**Diagnosis**: Round-robin doesn't account for scene complexity
**Fix**: Implement dynamic work-stealing (Issue #6)
```python
# Replace static partitioning with Queue-based work stealing
# See GPU_CODE_REVIEW.md section "Scheduling and Load Balancing"
```

---

## Performance Benchmarks

### Expected Throughput (Whisper medium model)
```
Single GPU:
  - Scene (2min video): ~10s processing
  - 100 scenes: ~1000s (16.7min)

Dual GPU (ideal):
  - Scene: ~10s (same)
  - 100 scenes: ~500s (8.3min)
  - Speedup: 2.0×

Dual GPU (actual):
  - Scene: ~10s
  - 100 scenes: ~550s (9.2min)
  - Speedup: 1.8×
  - Efficiency: 90% (good!)
```

### Bottleneck Analysis
If speedup < 1.5×, investigate:
1. Load imbalance (check worker scene counts)
2. I/O contention (check disk usage)
3. Memory bandwidth (check RAM usage)
4. Model loading overhead (check startup time)

---

## Reporting Results

### Test Report Template
```markdown
# Dual-GPU Stress Test Results

**Date**: YYYY-MM-DD
**Hardware**: 2× Intel Arc A770, i5-13600K
**Software**: Python 3.X, PyTorch X.X, IPEX X.X
**Patches Applied**: #1 (GPU affinity), #2 (VRAM), #3 (checkpointing)

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| Smoke Test | ✅ PASS | 2 workers, 2 GPUs detected |
| GPU Affinity | ✅ PASS | ZE_AFFINITY_MASK=0,1 verified |
| VRAM Safety | ✅ PASS | No OOM, 2 VRAM checks |
| Load Balancing | ✅ PASS | Ratio: 0.95, Speedup: 1.9× |
| Checkpointing | ✅ PASS | 10 .done markers, resume OK |
| Memory Leaks | ✅ PASS | Growth: +32% (acceptable) |
| Determinism | ⏭️ SKIP | --seed not implemented yet |

## Performance
- **Throughput**: 100 scenes in 9.2min
- **Speedup**: 1.8× (90% efficiency)
- **VRAM Usage**: Peak 3.2GB per GPU

## Issues Found
- None (all tests passed)

## Recommendations
- ✅ Production ready for dual-GPU deployment
- Consider dynamic work-stealing for better load balance
```

---

## Automated CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/gpu-stress-test.yml
name: GPU Stress Test

on:
  push:
    branches: [main]
  pull_request:

jobs:
  gpu-test:
    runs-on: [self-hosted, intel-arc]  # Custom runner with Arc GPUs

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest

    - name: Run smoke test
      run: pytest tests/test_dual_gpu_stress.py::test_smoke_dual_gpu -v

    - name: Run GPU affinity test
      run: pytest tests/test_dual_gpu_stress.py::test_gpu_affinity_dual_workers -v

    - name: Run VRAM test
      run: pytest tests/test_dual_gpu_stress.py::test_vram_utilization -v

    - name: Upload logs
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: gpu-test-logs
        path: |
          *.log
          scenes/.*.done
```

---

## Next Steps After Tests Pass

1. **Deploy to production**:
   - Run full 1000-scene batch
   - Monitor for 24 hours
   - Validate output quality

2. **Optimize further**:
   - Implement dynamic work-stealing (Issue #6)
   - Add per-worker metrics dashboard
   - Fine-tune model sizing heuristics

3. **Scale testing**:
   - Test with 4× GPUs (if available)
   - Test mixed workloads (subtitles + faces)
   - Test edge cases (corrupted videos, very long files)

---

**END OF STRESS TEST PLAN**
