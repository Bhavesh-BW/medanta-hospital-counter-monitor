"""RTSP Stream Frame Capture Helper.

Captures frames at a fixed interval (default: 1 frame/sec for 200 seconds)
directly from the live camera stream and saves them as timestamped JPEG images
ready for annotation in AnyLabeling.
"""

import argparse
import datetime
import os
import pathlib
import sys
import time

# Ensure OpenCV uses TCP for reliable RTSP streaming without dropped packets
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
import cv2

DEFAULT_RTSP_URL = os.environ.get('RTSP_URL', 'rtsp://124.123.89.162:31554/live/recep1')
DEFAULT_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / 'training_samples'


def capture_stream_samples(
    stream_url: str = DEFAULT_RTSP_URL,
    total_seconds: int = 200,
    interval_seconds: float = 1.0,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT_DIR,
    prefix: str = "sample"
) -> int:
    """Captures 1 frame per interval from an RTSP stream for a given duration.

    Args:
        stream_url: RTSP URL of the camera stream.
        total_seconds: Total duration to capture in seconds (default: 200).
        interval_seconds: Interval between captured frames (default: 1.0s).
        output_dir: Destination folder where JPEG screenshots are saved.
        prefix: Filename prefix for saved frames.

    Returns:
        Total number of successfully saved frames.
    """
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" 📹 RTSP Frame Capture Tool for Dataset Labeling")
    print("=" * 65)
    print(f" Source Stream   : {stream_url}")
    print(f" Target Duration : {total_seconds} seconds")
    print(f" Capture Rate    : 1 frame every {interval_seconds}s (approx {int(total_seconds / interval_seconds)} frames)")
    print(f" Save Directory  : {out_path.resolve()}")
    print("=" * 65)
    print("Connecting to RTSP stream, please wait...")

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ ERROR: Unable to open RTSP stream at: {stream_url}")
        print("Please check if the camera is online and network is reachable.")
        return 0

    # Read a test frame to ensure stream is active and get resolution
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ ERROR: Stream opened but failed to retrieve initial frame.")
        cap.release()
        return 0

    h, w = test_frame.shape[:2]
    print(f"✅ Stream connected successfully! Resolution: {w}x{h}")
    print("Starting capture session (press Ctrl+C to stop early)...\n")

    start_time = time.time()
    last_capture_time = 0.0
    saved_count = 0

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed >= total_seconds:
                print(f"\n⏱️ Reached duration limit ({total_seconds}s).")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                # Brief sleep before retrying to prevent busy waiting if packets drop
                time.sleep(0.05)
                continue

            # Check if 1 second has elapsed since last saved frame
            if (current_time - last_capture_time) >= interval_seconds:
                saved_count += 1
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"{prefix}_{saved_count:04d}_{now_str}.jpg"
                file_path = out_path / filename

                # Save as high-quality JPEG
                cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_capture_time = current_time

                percent = (elapsed / total_seconds) * 100
                print(f" [{saved_count:03d}] Saved: {filename} | Elapsed: {int(elapsed)}s/{total_seconds}s ({percent:.1f}%)")

    except KeyboardInterrupt:
        print("\n⚠️ Capture stopped manually by user.")
    finally:
        cap.release()

    print("\n" + "=" * 65)
    print(f"🎉 Capture Completed! Saved {saved_count} frames to:")
    print(f"   {out_path.resolve()}")
    print("=" * 65)
    return saved_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Capture frames from RTSP stream for model dataset training.")
    parser.add_argument('--stream', type=str, default=DEFAULT_RTSP_URL, help="RTSP stream URL (default: recep1)")
    parser.add_argument('--duration', type=int, default=200, help="Total capture time in seconds (default: 200)")
    parser.add_argument('--interval', type=float, default=1.0, help="Interval in seconds between frames (default: 1.0)")
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output folder for images")
    parser.add_argument('--prefix', type=str, default='sample', help="Prefix for image filenames")

    args = parser.parse_args()
    capture_stream_samples(
        stream_url=args.stream,
        total_seconds=args.duration,
        interval_seconds=args.interval,
        output_dir=args.output,
        prefix=args.prefix
    )

