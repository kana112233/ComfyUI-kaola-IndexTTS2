
import sys
import os
import traceback

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Testing imports from: {current_dir}")

def test_import(module_name, import_cmd):
    try:
        print(f"Testing: {module_name}...")
        exec(import_cmd)
        print(f"   SUCCESS: {module_name}")
        return True
    except ImportError as e:
        print(f"   FAILURE: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"   ERROR: {module_name} - {e}")
        traceback.print_exc()
        return False

# 1. Test kaola_indextts base
if not test_import("kaola_indextts", "import kaola_indextts"):
    sys.exit(1)

# 1.5 Test transformers_modeling_utils specifically for tensorflow avoidance
if not test_import("transformers_modeling_utils", "import kaola_indextts.gpt.transformers_modeling_utils as tmu; print(f'   Checked LOSS_MAPPING: {type(tmu.LOSS_MAPPING)}')"):
    sys.exit(1)

# 2. Test transformers_generation_utils specifically as it has the most external deps
utils_path = "kaola_indextts.gpt.transformers_generation_utils"
if not test_import(utils_path, f"import {utils_path}"):
    print("\nXXX Fatal Error: transformers_generation_utils failed to load even with patches.")
    sys.exit(1)

# 2.5 Test maskgct_utils specifically for protobuf/tensorflow avoidance
maskgct_path = "kaola_indextts.utils.maskgct_utils"
if not test_import(maskgct_path, f"import {maskgct_path}"):
    print("\nXXX Fatal Error: maskgct_utils failed to load. Transformers import shim failed.")
    sys.exit(1)

# 3. Test infer_v2 which depends on everything
if not test_import("IndexTTS2", "from kaola_indextts.infer_v2 import IndexTTS2"):
    sys.exit(1)

print("\nALL COMPREHENSIVE TESTS PASSED!")
