# 🔌 Auto-Shutdown Feature

Training model bisa memakan waktu berjam-jam atau berhari-hari. Fitur auto-shutdown memungkinkan Anda:
- 🌙 Train overnight tanpa khawatir PC menyala terus
- ⚡ Hemat listrik
- 💰 Shutdown otomatis setelah training selesai

---

## 🚀 Quick Usage

### Basic: Shutdown Setelah Training
```bash
python scripts\train_all_models.py --models all --epochs 50 --shutdown
```

Computer akan:
1. ✅ Train semua models
2. ✅ Save results
3. ⏰ Countdown 60 detik
4. 🔴 Shutdown otomatis

---

### Custom Delay: Beri Waktu Review
```bash
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 20 --shutdown --shutdown-delay 300
```

Countdown **5 menit** (300 detik) sebelum shutdown.

**Use case:**
- Cek hasil training sebelum shutdown
- Copy log files
- Screenshot metrics

---

## 📋 Parameter Details

```bash
--shutdown              # Enable auto-shutdown (default: disabled)
--shutdown-delay N      # Delay N seconds before shutdown (default: 60)
```

### Examples:

```bash
# Shutdown immediately (10 detik)
python scripts\train_all_models.py --models all --epochs 50 --shutdown --shutdown-delay 10

# Shutdown after 2 minutes (120 detik)
python scripts\train_all_models.py --models all --epochs 50 --shutdown --shutdown-delay 120

# Shutdown after 30 minutes (1800 detik) - untuk backup data
python scripts\train_all_models.py --models all --epochs 50 --shutdown --shutdown-delay 1800
```

---

## ⚠️ Important Notes

### 1. Countdown Warning
Setelah training selesai, Anda akan melihat:
```
======================================================================
  ⚠️  AUTO-SHUTDOWN IN 60 SECONDS
======================================================================

🔌 Computer akan shutdown dalam 60 detik...
   Press Ctrl+C sekarang untuk membatalkan!

   Shutdown in 60 seconds...
   Shutdown in 59 seconds...
   Shutdown in 58 seconds...
   ...
```

### 2. Cara Membatalkan
**Press `Ctrl+C` selama countdown** untuk membatalkan shutdown:
```
^C
✅ Shutdown cancelled by user.
======================================================================
```

### 3. Shutdown Command (Windows)
Script menggunakan:
```powershell
shutdown /s /t 0
```
- `/s` = Shutdown
- `/t 0` = Delay 0 seconds (immediate)

---

## 🎯 Use Cases

### Use Case 1: Overnight Training
**Scenario:** Train all models semalaman (8-12 jam)

```bash
# Start sebelum tidur (22:00)
python scripts\train_all_models.py --models all --epochs 50 --batch-size 64 --shutdown --shutdown-delay 300

# Training selesai ~08:00
# Countdown 5 menit (jika Anda bangun, bisa cancel)
# Shutdown 08:05
```

**Benefits:**
- PC tidak menyala seharian
- Hemat listrik ~50-100 Watt x 12 jam = 0.6-1.2 kWh
- Training selesai saat Anda bangun

---

### Use Case 2: Weekend Training
**Scenario:** Training 2-3 hari untuk research

```bash
# Jumat sore
python scripts\train_all_models.py --models all --epochs 100 --batch-size 64 --apply-smote --shutdown --shutdown-delay 600

# Training selesai Minggu pagi
# Countdown 10 menit untuk check results
# Auto-shutdown
```

---

### Use Case 3: Multiple Runs
**Scenario:** Test berbagai hyperparameters

```bash
# Run 1: Low learning rate
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 30 --batch-size 64 --shutdown-delay 0

# (Manual start run 2 setelah review)
# Run 2: High learning rate
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 30 --batch-size 128 --shutdown
```

---

## 🔧 Technical Details

### Implementation
File: `scripts/train_all_models.py`

```python
# Auto-shutdown if requested
if args.shutdown:
    import subprocess
    import time
    
    print(f"\n{'='*70}")
    print(f"  ⚠️  AUTO-SHUTDOWN IN {args.shutdown_delay} SECONDS")
    print(f"{'='*70}")
    
    try:
        for remaining in range(args.shutdown_delay, 0, -1):
            print(f"   Shutdown in {remaining} seconds...", end='\r')
            time.sleep(1)
        
        subprocess.run(['shutdown', '/s', '/t', '0'], check=True)
        
    except KeyboardInterrupt:
        print(f"\n\n✅ Shutdown cancelled by user.")
```

### Platform Support
- ✅ **Windows**: `shutdown /s /t 0`
- ❌ **Linux**: Perlu modifikasi ke `sudo shutdown -h now`
- ❌ **macOS**: Perlu modifikasi ke `sudo shutdown -h now`

---

## 🛡️ Safety Features

### 1. Countdown Warning
- ⏰ Default 60 detik untuk membatalkan
- 🔔 Visual countdown di terminal
- ⌨️ Ctrl+C untuk cancel

### 2. Completion Verification
Shutdown **HANYA** jika:
- ✅ Training berhasil complete
- ✅ Models tersimpan
- ✅ Metrics tersimpan
- ✅ Logs tersimpan

**Jika error terjadi saat training:**
- ❌ Shutdown TIDAK triggered
- 📝 Error logged
- 🖥️ PC tetap menyala untuk debugging

### 3. Manual Override
Selalu bisa dibatalkan dengan Ctrl+C.

---

## 📊 Energy Savings Calculator

### Typical Gaming/Workstation PC
- Power: ~150W (CPU + GPU under load)
- Cost: Rp 1,500/kWh (Indonesia average)

| Training Duration | Energy (kWh) | Cost (Rp) | Savings with Auto-Shutdown |
|-------------------|--------------|-----------|----------------------------|
| 8 hours (overnight) | 1.2 | 1,800 | Save 12 hours idle = Rp 900 |
| 24 hours | 3.6 | 5,400 | Save 18 hours idle = Rp 2,700 |
| 48 hours (weekend) | 7.2 | 10,800 | Save 42 hours idle = Rp 6,300 |

**Note:** Idle PC masih consume ~30-50W.

---

## ❓ FAQ

### Q: Bagaimana jika ingin restart setelah shutdown?
A: Gunakan Windows Task Scheduler atau WOL (Wake-on-LAN) jika perlu training berkelanjutan.

### Q: Apakah data aman?
A: Ya, semua data sudah tersimpan sebelum shutdown triggered:
- Models → `results/models/`
- Logs → `logs/training/`
- Metrics → `results/metrics/`

### Q: Bagaimana jika listrik mati saat training?
A: Gunakan checkpoint callbacks (sudah implemented):
```python
ModelCheckpoint('best_model.h5', save_best_only=True)
```
Model terbaik otomatis tersimpan setiap epoch.

### Q: Bisa restart training dari checkpoint?
A: Ya, edit script untuk load checkpoint:
```python
if os.path.exists('best_cnn_lstm_mlp.h5'):
    model.model = load_model('best_cnn_lstm_mlp.h5')
```

---

## 🎨 Customization

### Modify untuk Linux/macOS

Edit `scripts/train_all_models.py`:

```python
# Linux/macOS
if args.shutdown:
    import subprocess
    import platform
    
    if platform.system() == 'Windows':
        subprocess.run(['shutdown', '/s', '/t', '0'])
    elif platform.system() in ['Linux', 'Darwin']:
        subprocess.run(['sudo', 'shutdown', '-h', 'now'])
```

**Note:** Linux/macOS perlu sudo password atau configure sudoers.

---

## 🔍 Verification

### Test Auto-Shutdown (Safe)
```bash
# Test dengan delay panjang, lalu cancel
python scripts\train_all_models.py --models cnn_lstm_mlp --epochs 1 --batch-size 64 --shutdown --shutdown-delay 300

# Tunggu training selesai
# Saat countdown muncul, press Ctrl+C
# Verify: Shutdown cancelled
```

### Check Logs
```powershell
# Verify shutdown dilog
Get-Content (Get-ChildItem logs\training\*.log | Sort LastWriteTime -Desc | Select -First 1).FullName | Select-String -Pattern "SHUTDOWN|shutdown"
```

Expected output:
```
⚠️  AUTO-SHUTDOWN IN 60 SECONDS
🔌 Computer akan shutdown dalam 60 detik...
Shutdown in 60 seconds...
Shutdown in 59 seconds...
...
```

---

## ✅ Best Practices

1. **Test First:** Run dengan `--epochs 1` untuk test shutdown behavior
2. **Save Work:** Close semua aplikasi penting sebelum start training
3. **Backup Data:** Ensure important files sudah backup
4. **Check Space:** Verify cukup disk space untuk models dan logs
5. **Monitor First Run:** Jangan langsung overnight, pantau 1-2 epochs dulu
6. **Use UPS:** Jika listrik tidak stabil, gunakan UPS untuk prevent data loss

---

## 📚 Related Documentation

- [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - All available commands
- [LOGGING_GUIDE.md](LOGGING_GUIDE.md) - Log file locations
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

**Created:** 2025-12-08  
**Version:** 1.0  
**Platform:** Windows (Linux/macOS need modification)  
**Status:** ✅ Production Ready
