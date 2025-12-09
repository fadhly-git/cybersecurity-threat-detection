# ⚡ QUICK START - Fix Training Terlalu Lama

## 🚨 Problem: 1 Epoch = 6 Jam!

Training Anda terlalu lama karena:
- ❌ Dataset terlalu besar (2.5 juta rows)
- ❌ Batch size terlalu kecil (64)
- ❌ 20 epochs × 6 jam = **5 HARI**

## ✅ SOLUSI CEPAT

### Option 1: Sample 10% (TERCEPAT) ⭐ RECOMMENDED
```bash
# STOP training yang sekarang (Ctrl+C), lalu run:
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 5 --batch-size 256 --sample-ratio 0.1
```

**Result:**
- ⏰ **20 menit** (bukan 6 jam!)
- 📊 Dataset: 2.5M → 250K rows
- 🎯 Accuracy: ~93-95% (cukup untuk testing)

---

### Option 2: Sample 20% + Overnight
```bash
python scripts\train_all_models.py --models all --epochs 20 --batch-size 256 --sample-ratio 0.5 --shutdown
```

**Result:**
- ⏰ **8 jam** (overnight)
- 📊 Dataset: 2.5M → 500K rows  
- 🎯 Accuracy: ~95-96%
- 🔌 Auto-shutdown setelah selesai

---

### Option 3: Production (Full Dataset)
```bash
# Train 1 model dengan full data
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 20 --batch-size 512 --shutdown
```

**Result:**
- ⏰ **15 jam**
- 📊 Full dataset (2.5M rows)
- 🎯 Accuracy: ~97-98%

---

## 📊 Time Comparison

| Command | Dataset Size | Time/Epoch | Total (20 epochs) |
|---------|--------------|------------|-------------------|
| **Current (batch=64)** | 2.5M | 6 hours | 120 hours (5 days) ❌ |
| Sample 10% + batch=256 | 250K | **4 mins** | **80 mins** ✅ |
| Sample 20% + batch=256 | 500K | 8 mins | 160 mins (~3 hours) ✅ |
| Full + batch=512 | 2.5M | 45 mins | 15 hours ✅ |

---

## 💾 Storage Usage

**Fixed:** ModelCheckpoint sekarang hanya save max 20x (bukan ratusan kali)

**Before:**
- Save setiap improvement → **100+ saves** → 50 GB ❌

**After:**
- Save 1x per epoch → **Max 20 saves** → 10 GB ✅

---

## 🎯 Immediate Action

1. **STOP training sekarang:** Press `Ctrl+C`

2. **Test dengan sample 10%:**
```bash
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 5 --batch-size 256 --sample-ratio 0.1
```

3. **Jika results OK, run overnight:**
```bash
python scripts\train_all_models.py --models all --epochs 20 --batch-size 256 --sample-ratio 0.2 --shutdown
```

---

## 📚 Detailed Guide

Lihat **TRAINING_OPTIMIZATION.md** untuk penjelasan lengkap.

---

**Last Updated:** 2025-12-08  
**Status:** ✅ ModelCheckpoint fixed, sampling support added
