#!/usr/bin/env python3
"""
测试Flash Attention支持
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import verify_flash_attention, configure_flash_attention


def test_flash_attention():
    """测试Flash Attention支持"""
    print("=== Flash Attention 支持测试 ===")

    # 配置Flash Attention
    print("1. 配置Flash Attention...")
    configure_flash_attention(enable=True, disable_math_fallback=False)
    print("   ✓ Flash Attention配置完成")

    # 验证Flash Attention
    print("\n2. 验证Flash Attention支持...")
    try:
        info = verify_flash_attention(force=False, verbose=True)
        if info.available:
            print("   ✓ Flash Attention 2 可用")
            return True
        else:
            print("   ✗ Flash Attention 2 不可用")
            return False
    except Exception as e:
        print(f"   ✗ 验证失败: {e}")
        return False


def test_forced_flash_attention():
    """测试强制Flash Attention"""
    print("\n=== 强制Flash Attention测试 ===")

    try:
        info = verify_flash_attention(force=True, verbose=True)
        print("   ✓ 强制Flash Attention通过")
        return True
    except Exception as e:
        print(f"   ✗ 强制Flash Attention失败: {e}")
        return False


def main():
    """主测试函数"""
    print("Flash Attention支持测试\n")

    # 测试基本支持
    basic_test = test_flash_attention()

    # 测试强制模式
    forced_test = test_forced_flash_attention()

    print(f"\n=== 测试结果 ===")
    print(f"基本支持测试: {'通过' if basic_test else '失败'}")
    print(f"强制模式测试: {'通过' if forced_test else '失败'}")

    if basic_test and forced_test:
        print("🎉 Flash Attention 2 完全支持！")
        return True
    else:
        print("❌ Flash Attention 2 支持存在问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
