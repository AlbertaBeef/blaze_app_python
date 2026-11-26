# Implementation plan twnty thrid november

> Saved on 23 November 2026 at the user's request. The misspelling “twnty” is kept exactly as requested in the title.



## Goal and boundaries

Add a continuous, Pi-camera-driven mouse controller to the Hailo-8L hand-landmark application: move the pointer from the index-finger fingertip and produce one left-click per intentional thumb–index pinch. Keep this work in `~/hand-tracker-mouse`; leave the original upstream checkout and the separate `barehands` application untouched. Keep model weights local and out of Git.

## Verified starting point

- This Pi is a Raspberry Pi 5 running HailoRT. `/dev/hailo0` is present, `hailortcli scan` detects `0001:01:00.0`, and this account can use the accelerator.
- Both candidate Lite HEFs are explicitly compiled for Hailo-8L. `hailortcli parse-hef --parse-vstreams` reports the 192x192x3 Lite palm detector and 224x224x3 Lite landmark model. A real, one-second `hailortcli run` succeeded for each: palm 162 frames at 161.80 FPS; landmarks 298 frames at 296.14 FPS. These are model smoke-test rates on synthetic CLI input, not end-to-end camera, application, or pointer latency measurements.
- The upstream app already has a BlazePalm → 21-landmark hand inference path and can decode landmarks into the camera image. The user has independently confirmed a live Pi Camera inference test works. The copied `blaze_hailo/blaze_detect_live.py` retains the local CSI-camera `rpicam-vid`/YUV420 adaptation; the upstream checkout has a different uncommitted camera change and neither checkout's local edits are to be confused with a clean upstream checkout.
- The locally used models in the upstream checkout include `blaze_hailo/models/palm_detection_lite.hef` and `hand_landmark_lite.hef`. Upstream Git does not include them, and model weights must remain ignored locally, not committed.
- The current path defaults to live camera even with `--withoutview`; it still calls `cv2.waitKey`, so a headless/background run needs correct non-window input handling and a reliable stop path. Camera selection must be explicit: the copied camera initializer presently always starts Pi CSI with `PipeCamera` and does not actually honor a requested USB device. Do not describe USB failover as working until it is implemented and tested.
- The active desktop is **LXDE on labwc/Wayland**, not an X11-only desktop. A Wayland session and a separate TTY are active; their environment is not inherited by this agent. `/dev/uinput` is mode 0600, owned by root, with no `uinput` group. Therefore, X11-only pointer APIs, untested Wayland injection, a privileged system-wide service, and broad device-permission changes are not acceptable assumptions.
- `barehands.service` is a distinct local server. Avoid editing or restarting it; avoid opening competing camera streams while it uses a camera.

## Repository and Git status

- Separate working tree: `/home/carlito-cl/hand-tracker-mouse`.
- `main` preserves the cloned upstream snapshot. `upstream` points to `https://github.com/AlbertaBeef/blaze_app_python.git`.
- Current work branch: `feature/hailo-hand-mouse`. Changes made in the feature working tree cannot alter `/home/carlito-cl/Documents/archive/blaze_app_python`.
- Keep Hailo weights, image captures, profiling CSVs, Python environments, and runtime logs out of version control. Do not make or configure a GitHub repository, push, or expose model weights without the user explicitly selecting a remote, repository name, visibility, and appropriate Git identity/credentials.

## Implementation sequence

### 1. Capture the behavior with tests first

Add a small, GUI/Hailo-independent controller module and `pytest` unit tests. Use synthetic 21-point landmark arrays, timestamps, and a fake mouse sink to test: screen-edge clamping, the chosen mirrored/unmirrored orientation, low-confidence or missing hands leaving the pointer and button safe, smoothed movement, pinch thresholds with separate press/release hysteresis, holding a pinch producing at most one click, releasing then pinching again producing a second click, and short landmark dropouts not clicking. Run the specific tests red before implementing each behavior and run the full suite after each vertical feature slice. Keep the mouse sink injectable so tests never control the real pointer.

### 2. Map landmarks to the screen

- Use the index-finger tip (MediaPipe hand landmark 8) for pointer position.
- Normalize coordinates against the actual camera frame and a configurable, documented tracking rectangle; clamp to the active display's real pixel geometry. Make the mirrored-preview setting explicit and test each orientation against sample landmark arrays.
- Smooth movement with a tunable, time-aware low-pass filter/dead zone to reduce camera jitter without making the cursor feel laggy. Reacquire safely when a hand first appears; do not sweep the mouse across the screen from an old point.
- Ignore missing or low-confidence landmarks. Loss of a valid hand must always release any held mouse button and clear gesture state.

### 3. Detect a pinch and emit a click

Measure thumb tip (landmark 4) to index tip (landmark 8) relative to palm scale (initially use the wrist-to-middle-finger-MCP landmarks 0–9, then tune against actual camera output). Separate pinch-down and pinch-up thresholds to add hysteresis. Require a brief configurable dwell/confirmation, trigger a primary-button down/up pair once on the transition into a pinch, and require a real release before arming another click. Hand loss, inference exceptions, application shutdown, and SIGTERM must release any outstanding button. Do not treat a continuing pinch as repeated clicks. Tune distances and dwell with the user in front of the live preview, not guessed model coordinates.

### 4. Add the runtime behind explicit controls

Keep capture, Hailo inference/post-processing, gesture logic, and mouse output as separate seams. Preserve existing preview and detector-only behavior. Add clear command-line options for a non-preview continuous mode, enabling/disabling mouse control, camera choice (CSI or a specifically selected USB device), model paths, screen mapping, mirrored input, and pinch/filter tuning. Validate configuration and required device/file access before activating mouse output. Log the actual selected camera, HEF paths, display session, and clean stop reason. Handle SIGINT/SIGTERM and recover or exit cleanly on camera loss or inference error; never leave the left button down. Do not keep a hidden blocking OpenCV keyboard wait in headless mode.

### 5. Prove safe Wayland pointer integration before service/autostart

First probe which pointer injection options the active labwc session actually permits. Prefer a supported, session-scoped Wayland virtual-pointer/input protocol if the installed compositor exposes it; test pointer motion and button down/up in both native Wayland and XWayland windows. An XTest-only implementation is not sufficient for native Wayland. If the compositor offers no safe user-session mechanism, stop and ask before changing udev, granting `uinput`, installing privileged helpers, or proposing broader permissions. Never run the vision or mouse process as root and never install it as a system-level root service.

### 6. Integrate user-session autostart, only after manual acceptance

Once command-line pointer control is confirmed, create a `systemd --user` unit with an explicit `WorkingDirectory`, predictable Python/virtual-environment interpreter and Hailo library environment, required camera dependency, safe restart/stop behavior, and logs available via `journalctl --user`. Start it from the user's labwc graphical-session startup mechanism so `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, and the compositor are available. Do not enable startup by default until the user has watched the safety tests and agreed: an automatic pointer controller can interfere with normal keyboard/mouse use. It should work only in the logged-in graphical session; it cannot operate the desktop at the lock screen. Add documented `start`, `stop`, `disable`, `status`, and recovery/rollback instructions. Do not run `barehands`' server differently or change its camera ownership without consent.

### 7. Document and run the end-to-end acceptance checks

- Unit tests pass on CPU with no camera, Hailo device, GUI, or real mouse activity.
- Existing Pi-camera hand detection and 21 landmarks still work; the pointer follows landmark 8 smoothly and clamps correctly at all four screen edges.
- One held pinch gives exactly one left-click; releasing and pinching again gives exactly one additional click. Jitter, low confidence, brief hand loss, and shutdown do not cause a false click or stuck button.
- CSI and USB camera selection each either operate as selected or fail with a clear explanation; do not claim automatic backup until a tested handoff exists.
- The correctly authorized labwc session can move the pointer and click in real native-Wayland and XWayland targets. Start/stop/restart, login, logout, service logs, SIGTERM cleanup, and emergency disable are verified on the actual Pi.
- `pytest` and syntax/static checks pass. Review `git status`, the complete diff, and tracked file list to verify no model weights, credentials, captured images, environments, or unrelated upstream files entered the commit.

## Implementation decisions that remain unverified

1. Whether the installed labwc/wlroots build exposes a usable virtual-pointer protocol to a normal Wayland client; actual pointer injection must prove it.
2. Whether this model's selected index-tip and thumb-tip coordinates are stable enough at the actual Pi Camera distance and image mirroring; collect/tune live landmark traces with the user.
3. Which USB video node/pipeline provides the user's working backup camera and how CSI/USB camera selection should be presented.
4. Whether normal login, display rotation/resolution changes, and compositor startup/shutdown permit reliable user-unit autostart and pointer mapping. Verify rather than assuming.
