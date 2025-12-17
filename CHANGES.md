# Version Changes and Updates

## Version 1.0.0 - December 16, 2025

### Major Changes
- Fixed path compatibility issues between Windows and Linux environments for .pt file loading
- Added robust error handling and CPU fallback for CUDA compatibility issues
- Updated library compatibility for newer versions of timm and PyTorch

### Files Modified
1. `ssast-main/src/dataloader_pt_reader.py`
   - Added regex-based path processing to handle Windows and Unix style paths
   - Implemented file validation before loading to avoid IndexError
   - Added fallback mechanism for missing files

2. `ssast-main/src/run.py`
   - Added CUDA error handling with automatic CPU fallback
   - Fixed model parameter validation for pretraining
   - Updated JSON mapping logic for training/validation datasets

3. `ssast-main/src/models/ast_models.py`
   - Added timm library version compatibility
   - Added fallback model names for newer timm versions
   - Removed strict version assertion

4. `ssast-main/src/dataloader_pt_reader.py`
   - Enhanced path extraction logic
   - Added validation for file existence
   - Improved error messages for debugging

### Features Added
- Cross-platform path handling
- Automatic CPU fallback when CUDA fails
- Library version compatibility
- Improved error reporting and debugging

### Issues Resolved
- FileNotFoundError when loading .pt files with Windows-style paths
- CUDA kernel compatibility issues
- Model parameter validation errors
- Timm library version incompatibility

### Testing
- Local CPU pretraining test verified
- Path mapping validation confirmed
- .pt file loading functionality tested
- Model creation and forward pass confirmed