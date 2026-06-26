# benchmark.py
import time
import requests
import numpy as np

print("🚀 Starting 50ms benchmark (100 requests)...")
latencies = []

for i in range(100):
    # Generate random 82-channel sensor reading
    sample = np.random.randn(82).tolist()
    
    start = time.perf_counter()
    try:
        response = requests.post(
            "http://localhost:8000/stream",
            json={"data": [sample]},
            timeout=5
        )
        end = time.perf_counter()
        
        ms = (end - start) * 1000
        latencies.append(ms)
        
        if response.status_code != 200:
            print(f"❌ Request {i} failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Request {i} error: {e}")

# Calculate stats
if latencies:
    avg = sum(latencies) / len(latencies)
    max_l = max(latencies)
    min_l = min(latencies)
    
    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS")
    print("="*50)
    print(f"✅ Total requests: {len(latencies)}")
    print(f"✅ Average latency: {avg:.2f} ms")
    print(f"✅ Max latency:     {max_l:.2f} ms")
    print(f"✅ Min latency:     {min_l:.2f} ms")
    print("="*50)
    
    if avg < 50:
        print("🎉 PASS! Your inference pipeline meets the 50ms constraint!")
    else:
        print(f"⚠️  FAIL. Average is {avg:.2f} ms. Need to optimize further.")
else:
    print("❌ No successful requests. Check if server is running.")