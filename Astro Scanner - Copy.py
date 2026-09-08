import base64
import time
import cv2
import numpy as np
import obsws_python as obs

# Configuration
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = ""
SOURCE_NAME = ""

# Filtering parameters
MIN_RELATIVE_SPEED = 0.005  
MAX_RELATIVE_SPEED = 10.0     
MOTION_CLEAR_DELAY = 10.0 
REQUIRED_CONSECUTIVE_FRAMES = 3

def decode_base64_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def main():
    try:
        client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
        print("Connected to OBS WebSocket successfully.")
    except Exception as e:
        print(f"Failed to connect to OBS: {e}")
        return

    tracked_objects = {}  
    next_object_id = 0
    is_recording = False
    last_motion_time = 0
    
    last_frame_time = 0
    TARGET_FPS = 15  

    prev_gray = None

    print("Starting optimized night-sky monitoring loop... Press Ctrl+C to exit.")

    while True:
        try:
            current_time = time.time()
            if current_time - last_frame_time < (1.0 / TARGET_FPS):
                time.sleep(0.005)
                continue
            last_frame_time = current_time

            response = client.get_source_screenshot(
                name=SOURCE_NAME, 
                img_format="jpeg", 
                width=960, 
                height=600, 
                quality=60
            )
            frame = decode_base64_image(response.image_data)

            if frame is None:
                print("Warning: Received empty frame from OBS.")
                time.sleep(1)
                continue

            frame_height, frame_width = frame.shape[:2]
            max_pixel_jump = int(frame_width * MAX_RELATIVE_SPEED)

            # Preprocess
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            # 1. Absolute Frame Differencing (Instead of flawed background subtraction)
            # This ignores static background completely and only looks at changes between consecutive frames
            diff = cv2.absdiff(prev_gray, gray)
            
            # 2. Strict Intensity Thresholding: Only allow bright flashes/streaks (meteors/sats), ignore dim scintillation noise
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

            # Clean up minor isolated pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_tracked_objects = {}
            motion_detected = False

            for cnt in contours:
                if cv2.contourArea(cnt) < 15:  # Ignore tiny pixel blips
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                matched_id = None
                min_dist = float("inf")
                for obj_id, data in tracked_objects.items():
                    pcx, pcy = data['pos']
                    dist = np.hypot(cx - pcx, cy - pcy)
                    if dist < min_dist and dist < max_pixel_jump:  
                        min_dist = dist
                        matched_id = obj_id

                if matched_id is None:
                    matched_id = next_object_id
                    next_object_id += 1
                    streak = 0
                    prev_pos = None
                else:
                    prev_pos = tracked_objects[matched_id]['pos']
                    streak = tracked_objects[matched_id]['streak']

                if prev_pos is not None:
                    pcx, pcy = prev_pos
                    pixel_displacement = np.hypot(cx - pcx, cy - pcy)
                    relative_velocity = pixel_displacement / frame_width

                    if MIN_RELATIVE_SPEED <= relative_velocity <= MAX_RELATIVE_SPEED:
                        streak += 1
                    else:
                        streak = 0  
                else:
                    streak = 0  

                current_tracked_objects[matched_id] = {
                    'pos': (cx, cy),
                    'streak': streak,
                    'velocity': relative_velocity if prev_pos is not None else 0.0
                }

                if streak >= REQUIRED_CONSECUTIVE_FRAMES:
                    vel = current_tracked_objects[matched_id]['velocity']
                    print(f"Target trajectory locked! ID: {matched_id} (Streak: {streak} frames, Velocity: {vel:.4f} rel/frame)")
                    motion_detected = True

            tracked_objects = current_tracked_objects
            prev_gray = gray  .copy() # Update previous frame reference
            current_time = time.time()

            if motion_detected:
                last_motion_time = current_time
                if not is_recording:
                    try:
                        client.start_record()
                        is_recording = True
                        print("Confirmed trajectory — Started OBS Recording.")
                    except Exception as obs_err:
                        print(f"Error starting recording: {obs_err}")

            elif is_recording:
                if (current_time - last_motion_time) > MOTION_CLEAR_DELAY:
                    try:
                        client.stop_record()
                        is_recording = False
                        print("Target left frame — Stopped OBS Recording.")
                    except Exception as obs_err:
                        print(f"Error stopping recording: {obs_err}")

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("Exiting script...")
            if 'is_recording' in locals() and is_recording:
                try:
                    client.stop_record()
                    print("Stopped active recording on exit.")
                except:
                    pass
            break
        except Exception as err:
            print(f"Runtime error: {err}")
            time.sleep(2)

if __name__ == "__main__":
    main()
