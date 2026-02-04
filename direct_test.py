#!/usr/bin/env python3
"""
直接测试我们的修复，绕过数据准备阶段
"""

import sys
import os
import tempfile
from pathlib import Path
import torch
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_checkpoint_saving():
    """测试检查点保存功能"""
    print("测试检查点保存功能...")

    try:
        # 导入保存函数
        from litgpt.pretrain import save_checkpoint

        # 创建模拟对象
        fabric = Mock()
        fabric.global_rank = 0
        fabric.save = Mock()
        fabric.print = Mock()

        # 创建模型配置模拟
        config = Mock()
        config.to_dict = Mock(return_value={"test": "config"})

        # 创建模型模拟
        model = Mock()
        model.config = config

        # 创建状态字典
        state = {"model": model, "optimizer": Mock(), "iter_num": 100, "step_count": 50}

        # 在临时目录中测试
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint" / "lit_model.pth"

            # 调用保存函数
            save_checkpoint(fabric, state, None, checkpoint_path)

            # 验证调用是否正确
            fabric.save.assert_called_once_with(checkpoint_path, state)
            print("✅ 检查点保存功能正常工作")
            return True

    except Exception as e:
        print(f"❌ 检查点保存测试失败: {e}")
        return False


def test_csv_logging():
    """测试CSV日志记录功能"""
    print("\n测试CSV日志记录功能...")

    try:
        # 导入日志记录函数
        from litgpt.utils import choose_logger
        from lightning.pytorch.loggers import CSVLogger

        # 在临时目录中测试
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # 创建CSV日志记录器
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="direct_test", log_interval=1)

            # 验证是否创建了正确的日志记录器
            if isinstance(logger, CSVLogger):
                print("✅ CSV日志记录功能正常工作")
                return True
            else:
                print(f"❌ 期望CSVLogger，但得到了 {type(logger)}")
                return False

    except Exception as e:
        print(f"❌ CSV日志记录测试失败: {e}")
        return False


def test_directory_creation():
    """测试目录创建功能"""
    print("\n测试目录创建功能...")

    try:
        # 导入日志记录函数
        from litgpt.utils import choose_logger

        # 在临时目录中测试
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            logs_dir = out_dir / "logs" / "csv"

            # 创建日志记录器（应该会创建目录）
            logger = choose_logger(logger_name="csv", out_dir=out_dir, name="direct_test", log_interval=1)

            # 检查目录是否被创建
            if logs_dir.exists():
                print("✅ 目录创建功能正常工作")
                return True
            else:
                print("❌ 日志目录未被创建")
                return False

    except Exception as e:
        print(f"❌ 目录创建测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("直接测试我们的修复...\n")

    tests = [
        test_checkpoint_saving,
        test_csv_logging,
        test_directory_creation,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n最终结果: {passed}/{len(tests)} 个测试通过")

    if passed == len(tests):
        print("🎉 所有核心功能修复都已验证成功！")
        return True
    else:
        print("❌ 一些修复未能验证")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
