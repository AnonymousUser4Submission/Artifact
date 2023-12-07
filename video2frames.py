import cv2
import os

def convert_mp4_to_jpg(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video from the file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        # Read a frame
        success, frame = cap.read()

        # If a frame was read successfully, save it as a JPEG
        if success:
            cv2.imwrite(f'{output_folder}/frame_{frame_count:04d}.jpg', frame)
            frame_count += 1
        else:
            # If no frame was read (end of video), break the loop
            break

    # Release the video capture object
    cap.release()

# Usage
convert_mp4_to_jpg('video.mp4', 'frames')
