# BirdNet Background Extraction Guide

This guide explains how to run BirdNet feature extraction in the background so you can safely logout while the process continues.

## Quick Start

### Step 1: Start Background Extraction

```bash
cd /data/4joyson/bdnt
./start_background_extraction.sh
```

This will:
- Start extraction as a background process (survives logout)
- Extract Training Set: 5 classes in parallel (HT, JD, BV, WMW, MT)
- Then extract Validation Set: 3 classes in parallel (ME, PB, PB24)
- Log everything to separate files
- Estimated time: 6-18 hours

### Step 2: Monitor Progress (Optional)

**View all combined logs:**
```bash
tail -f /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log
```

**View specific class extraction:**
```bash
tail -f /data/4joyson/bdnt/birdnet/extraction_logs/training/HT.log
tail -f /data/4joyson/bdnt/birdnet/extraction_logs/validation/ME.log
```

**Check status summary:**
```bash
/data/4joyson/bdnt/check_extraction_status.sh
```

### Step 3: Logout (Optional)

Once the process starts, you can safely logout. The extraction will continue in the background:

```bash
logout
# or
exit
```

---

## File Organization

### Log Files Structure:
```
/data/4joyson/bdnt/birdnet/extraction_logs/
├── training/
│   ├── combined.log     # All training set output
│   ├── HT.log          # Class HT extraction log
│   ├── JD.log          # Class JD extraction log
│   ├── BV.log          # Class BV extraction log
│   ├── WMW.log         # Class WMW extraction log
│   └── MT.log          # Class MT extraction log
├── validation/
│   ├── combined.log     # All validation set output
│   ├── ME.log          # Class ME extraction log
│   ├── PB.log          # Class PB extraction log
│   └── PB24.log        # Class PB24 extraction log
└── status/
    ├── extraction.pid  # Process ID
    └── extraction_status.txt  # Completion status
```

### Extracted Embeddings:
```
/data/msc-proj/Training_Set/
├── HT/
│   ├── h1.wav
│   └── h1_BDnet.npy        ✓ New embedding (num_windows, 6522)
├── JD/
│   └── ...
└── ...

/data/msc-proj/Validation_Set_DSAI_2025_2026/
├── ME/
│   └── ...
├── PB/
│   └── ...
└── PB24/
    └── ...
```

---

## Monitoring Commands

### Real-time Progress (Training)
```bash
tail -f /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log
```

### Real-time Progress (Validation)
```bash
tail -f /data/4joyson/bdnt/birdnet/extraction_logs/validation/combined.log
```

### Quick Status Check
```bash
/data/4joyson/bdnt/check_extraction_status.sh
```

### Count Extracted Files
```bash
find /data/msc-proj -name "*_BDnet.npy" | wc -l
```

### Watch specific class
```bash
watch -n 5 "find /data/msc-proj/Training_Set/HT -name '*_BDnet.npy' | wc -l"
```

---

## Process Management

### Check if extraction is still running
```bash
ps aux | grep extract_embeddings
```

### View extraction process details
```bash
cat /data/4joyson/bdnt/birdnet/extraction_logs/status/extraction.pid
```

### View current completion status
```bash
cat /data/4joyson/bdnt/birdnet/extraction_logs/status/extraction_status.txt
```

---

## After Extraction Completes

Once all embeddings are extracted, you'll see:
- ✓ Message in both log files
- Status file updated to "COMPLETE"
- ~300-400 total `_BDnet.npy` files created

Then run training:

```bash
cd /data/4joyson/bdnt
python train.py --config-name train_birdnet +exp_name="BirdNet"
```

---

## Troubleshooting

### Check if process is still running
```bash
/data/4joyson/bdnt/check_extraction_status.sh
```

### Check specific class logs for errors
```bash
grep -i "error\|failed\|traceback" /data/4joyson/bdnt/birdnet/extraction_logs/training/HT.log
```

### View last 100 lines of training log
```bash
tail -100 /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log
```

### If extraction seems stuck
Check log timestamps to see which class is being processed:
```bash
tail -20 /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log
```

---

## Performance Notes

### Extraction Speed:
- **Single class**: 15-30 minutes per 1-hour audio file
- **5 parallel processes**: ~3-5x faster than sequential
- **Total time**: 6-18 hours for all 186 files

### Resource Usage:
- **CPU**: Uses all available cores (8 threads per process)
- **Memory**: ~2-4 GB peak
- **Disk I/O**: ~100-200 MB/s during peak

### Why Parallel is Important:
- **Sequential**: 30-90 hours
- **5 parallel processes**: 6-18 hours
- **Speedup**: 5-10x faster

---

## Summary

1. Start extraction: `./start_background_extraction.sh`
2. Logout anytime (process continues)
3. Monitor anytime: `./check_extraction_status.sh`
4. After complete, run training: `python train.py --config-name train_birdnet`

That's it! The extraction runs completely in the background with separate logs for Training and Validation sets.
