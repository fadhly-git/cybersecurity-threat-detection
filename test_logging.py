"""Test script untuk verifikasi logging system.

Jalankan dengan:
    python test_logging.py
    
Output akan masuk ke console DAN file logs/test/test_logging_TIMESTAMP.log
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import DualOutput


def main():
    # Setup logging
    Path('logs/test').mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test/test_logging_{timestamp}.log'
    
    print(f"Testing logging system...")
    print(f"Log file: {log_file}\n")
    
    # All output below will go to both console and file
    with DualOutput(log_file):
        print("="*70)
        print("  LOGGING SYSTEM TEST")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        print("\n✅ Testing print statements:")
        print("   - This is a normal print")
        print("   - This is another print")
        
        print("\n✅ Testing formatted output:")
        for i in range(5):
            print(f"   Iteration {i+1}/5: Processing...")
        
        print("\n✅ Testing error messages:")
        try:
            result = 10 / 0
        except Exception as e:
            print(f"   ❌ Error caught: {e}")
        
        print("\n✅ Testing multi-line output:")
        data = {
            'model': 'CNN-LSTM-MLP',
            'accuracy': 0.985,
            'f1_score': 0.972,
            'training_time': '45 minutes'
        }
        for key, value in data.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*70)
        print("  TEST COMPLETED")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    print(f"\n✅ Check log file: {log_file}")


if __name__ == '__main__':
    main()
