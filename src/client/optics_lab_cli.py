"""optic_lab_cli.py module contains a command line interface to run
remote recording of a video on the Raspberry Pi and generate data file."""

import logging
import argparse
import shutil

import rpyc
import cv2
import numpy as np

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the optics lab")
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        default="rpi32.local",
        help="The IP address of the Raspberry Pi",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="foo",
        help="The name of the file to save the video to",
    )
    parser.add_argument(
        "-t",
        "--record_time",
        type=int,
        default=30,
        help="The time to record the video for",
    )
    args = parser.parse_args()

    # Connect to the Raspberry Pi
    pi_connection: rpyc.Connection = rpyc.connect(
        "rpi32.local", 18862, config={"allow_pickle": True, "sync_request_timeout": 240}
    )

    pi_connection.root.configure_camera(awb_gains=(1, 2))
    pi_connection.root.run_camera_sample(record_time=args.record_time, filename="foo")

    remote_h264, remote_mp4 = pi_connection.root.open_file("foo")
    with open(f"{args.filename}.h264", "wb") as local_h264:
        shutil.copyfileobj(remote_h264, local_h264)
    with open(f"{args.filename}.mp4", "wb") as local_mp4:
        shutil.copyfileobj(remote_mp4, local_mp4)

    pi_connection.root.close_file()

    # Get raw data of the video
    capture = cv2.VideoCapture(f"{args.filename}.mp4", cv2.CAP_FFMPEG)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    mean_signal = np.zeros((num_frames, 3))

    # loop through the video
    count: int = 0
    region_of_interest: tuple[int, int, int, int] = (0, 0, 0, 0)
    while capture.isOpened():
        #'frame' is a normal numpy array of dimensions [height, width, 3], in order BGR
        ret, frame = capture.read()
        if not ret:
            break

        # display window for selection of ROI
        if count == 0:
            WINDOW_TEXT = (
                "Select ROI by dragging the mouse, "
                + "and press SPACE or ENTER once satisfied."
            )
            # ROI contains: [x, y, w, h] for selected rectangle
            region_of_interest = cv2.selectROI(WINDOW_TEXT, frame)
            cv2.destroyWindow(WINDOW_TEXT)
            print("Looping through video.")

        # calculate mean
        cropped_frame = frame[
            region_of_interest[1] : region_of_interest[1] + region_of_interest[3],
            region_of_interest[0] : region_of_interest[0] + region_of_interest[2],
            :,
        ]
        mean_signal[count, :] = np.mean(cropped_frame, axis=(0, 1))
        count += 1

    capture.release()

    # save to file in order R, G, B.
    OUTPUT_FILENAME = f"{args.filename}.txt"
    np.savetxt(OUTPUT_FILENAME, np.flip(mean_signal, 1))
    print("Data saved to '" + OUTPUT_FILENAME + "', fps = " + str(fps) + " frames/second")
