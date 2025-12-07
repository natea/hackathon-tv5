-0# LibGL Fix for OpenCV in Docker/DevContainer

## Problem

When running the Ginger backend in a Docker container or DevContainer, OpenCV (cv2) fails to import with the following error:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This occurs because OpenCV requires OpenGL libraries that are not installed by default in minimal Debian/Ubuntu containers.

## Root Cause

The `pipecat` library imports OpenCV (`cv2`) for WebRTC video processing in `pipecat/transports/smallwebrtc/transport.py`. OpenCV's Python bindings depend on `libGL.so.1` from the Mesa OpenGL library.

## Solution

Install the required system libraries:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### What gets installed

- `libgl1-mesa-glx` - OpenGL library (provides `libGL.so.1`)
- `libglib2.0-0` - GLib library (common dependency)
- Various X11/XCB libraries (transitive dependencies)

## Adding to DevContainer

To make this fix permanent, add the packages to the DevContainer configuration.

### Option 1: Dockerfile

Add to your Dockerfile:

```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

### Option 2: devcontainer.json

Add to your `devcontainer.json`:

```json
{
  "postCreateCommand": "sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0"
}
```

### Option 3: Features

Use the common-utils feature with additional packages:

```json
{
  "features": {
    "ghcr.io/devcontainers/features/common-utilities:2": {
      "installZsh": false,
      "configureZshAsDefaultShell": false,
      "installOhMyZsh": false,
      "upgradePackages": false
    }
  },
  "postCreateCommand": "sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0"
}
```

## Alternative: Headless OpenCV

If you don't need GUI features, you can use `opencv-python-headless` instead of `opencv-python`:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

This version doesn't require libGL but may lack some features.

## Verification

After installing, verify the fix:

```bash
python -c "import cv2; print(cv2.__version__)"
```

## Related Issues

- pydub warning about ffmpeg: Install `ffmpeg` if audio processing is needed
- Additional codec support: Install `libavcodec-extra` for more video formats
