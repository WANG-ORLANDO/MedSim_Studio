#!/usr/bin/env python3
"""验证 B 组开发环境"""
import sys

def check_module(name):
    try:
        __import__(name)
        print(f"  ✅ {name}")
        return True
    except ImportError:
        print(f"  ❌ {name} NOT FOUND")
        return False

print("B 组环境检查:")
print("核心依赖:")
check_module("numpy")
check_module("scipy")
check_module("skimage")
check_module("torch")
check_module("torchvision")
check_module("timm")

print("\nA 组接口测试:")
try:
    import requests
    r = requests.get("http://localhost:8000/api/v1/health")
    assert r.status_code == 200
    print("  ✅ Backend API 可达")
except Exception as e:
    print(f"  ❌ Backend API 不可达: {e}")

print("\n体积解码测试:")
try:
    import numpy as np
    import base64
    # 模拟解码
    vol = np.random.randn(32, 32, 32).astype(np.float32)
    b64 = base64.b64encode(vol.tobytes()).decode()
    decoded = np.frombuffer(base64.b64decode(b64), dtype='<f4').reshape((32, 32, 32))
    assert np.allclose(vol, decoded)
    print("  ✅ 体积解码逻辑正确")
except Exception as e:
    print(f"  ❌ 解码测试失败: {e}")

print("\n检查完成")
