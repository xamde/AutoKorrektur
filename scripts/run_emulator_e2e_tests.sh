#!/usr/bin/env bash
set -e

EMULATOR_BIN="/home/konrad/Android/Sdk/emulator/emulator"
AVD_NAME="Pixel_9_Pro_XL"

echo "=== AutoKorrektur: End-to-End Emulated UI Test Runner ==="

# 1. Check if an emulator is already attached
if adb get-state 2>/dev/null | grep -q "device"; then
    echo "[INFO] An active Android device/emulator is already attached."
else
    echo "[INFO] Starting emulator $AVD_NAME in headless mode..."
    $EMULATOR_BIN -avd "$AVD_NAME" -no-window -no-audio -gpu swiftshader_indirect > /dev/null 2>&1 &
    
    echo "[INFO] Waiting for emulator to boot..."
    adb wait-for-device
    while [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != "1" ]]; do
        sleep 2
    done
    echo "[INFO] Emulator booted successfully!"
fi

# 2. Setup port reverse for SDXL backend if needed
adb reverse tcp:8000 tcp:8000 2>/dev/null || true

# 3. Execute the FullEmulatedUiInferenceE2ETest
echo "[INFO] Executing FullEmulatedUiInferenceE2ETest..."
./gradlew :app:connectedDebugAndroidTest \
    -Pandroid.testInstrumentationRunnerArguments.class=de.konradvoelkel.android.autokorrektur.FullEmulatedUiInferenceE2ETest

echo "[INFO] Full E2E UI Test Suite Completed Successfully!"
