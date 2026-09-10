"""
Hand Gesture Virtual Mouse
---------------------------
Controls the system mouse cursor using webcam hand-tracking (MediaPipe Hands).

Install dependencies first:
    pip install opencv-python mediapipe pyautogui

Gestures:
    - Move index fingertip  -> move cursor
    - Pinch (thumb + index) -> left click
    - Pinch (thumb + middle)-> right click
    - Raise index + middle together, move hand up/down -> scroll

Press 'q' in the webcam window to quit.
"""

import time
import math
import cv2
import mediapipe as mp
import pyautogui

# =========================================================================
# CONFIGURATION — tune these values to change behavior. Keep them together
# so the whole system's "feel" can be adjusted from one place.
# =========================================================================

# --- Cursor smoothing ---
# Exponential smoothing factor applied every frame:
#   smooth = previous + (target - previous) * SMOOTHING_FACTOR
# Lower  -> cursor reacts faster but looks jittery.
# Higher -> cursor glides smoothly but feels laggy/slow to respond.
SMOOTHING_FACTOR = 0.55

# --- Screen mapping margin ---
# Fraction of the webcam frame (on each side) treated as "dead" margin,
# not mapped to the screen. This lets the user reach screen edges/corners
# without moving their finger to the physical edge of the camera frame.
# Larger margin -> smaller physical hand movement needed to cover the
# whole screen (more sensitive). Too large and you lose usable frame area.
FRAME_MARGIN = 0.15  # 0.15 = 15% margin on each side

# --- Click thresholds (in PIXELS, measured in the webcam frame) ---
# Distance between fingertip landmarks below which a pinch counts as a click.
# Smaller  -> fingers must be almost touching (harder to trigger accidentally,
#             but also harder to trigger on purpose).
# Larger   -> easier to trigger, but more prone to accidental clicks.
LEFT_CLICK_THRESHOLD = 30   # thumb tip <-> index tip
RIGHT_CLICK_THRESHOLD = 30  # thumb tip <-> middle tip

# --- Click cooldown ---
# Minimum seconds between two clicks of the SAME type. Prevents a single
# held pinch from firing dozens of clicks per second (once per frame).
CLICK_COOLDOWN = 0.3

# --- Scroll settings ---
# SCROLL_THRESHOLD: minimum vertical hand movement (in pixels, webcam frame)
# between frames before it counts as an intentional scroll. Filters out
# hand tremor / noise.
#   Increase -> less sensitive, requires bigger hand movement to scroll.
#   Decrease -> more sensitive, small movements trigger scroll.
SCROLL_THRESHOLD = 15

# SCROLL_SPEED: how many "scroll units" pyautogui.scroll() moves per
# qualifying frame of movement. Higher = faster scrolling per motion.
SCROLL_SPEED = 5

# --- MediaPipe detection confidence ---
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Landmark indices (MediaPipe Hands) ---
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6   # index finger's lower knuckle joint, used to check "raised"
MIDDLE_TIP = 12
MIDDLE_PIP = 10

# =========================================================================
# PYAUTOGUI SAFETY
# =========================================================================
# FAILSAFE stays ON (default True): if the real physical mouse is dragged
# to a screen corner (0,0), PyAutoGUI raises FailSafeException and stops
# the program. This is an emergency "kill switch" if gesture control
# misbehaves, so we deliberately do NOT disable it.
pyautogui.FAILSAFE = True
# Removes the small artificial delay pyautogui inserts after every call,
# since we already control pacing via our own smoothing/cooldown logic.
pyautogui.PAUSE = 0.0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


def clamp(value, min_value, max_value):
    """Restrict a value to the inclusive range [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def euclidean_distance(point_a, point_b):
    """Straight-line pixel distance between two (x, y) points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


class HandTracker:
    """Wraps MediaPipe Hands: detects landmarks and draws them on a frame."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr):
        """
        Runs hand detection on a BGR frame.
        Returns a list of (x_px, y_px) pixel coordinates for all 21
        landmarks of the first detected hand, or None if no hand is found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the skeleton overlay directly on the frame for visual feedback.
        self.mp_drawing.draw_landmarks(
            frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

        frame_height, frame_width = frame_bgr.shape[:2]
        landmark_points = [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks.landmark
        ]
        return landmark_points

    def close(self):
        self.hands.close()


class GestureRecognizer:
    """
    Turns raw landmark points into gesture events: click state and
    scroll-mode state. Keeps cooldown/timing logic self-contained.
    """

    def __init__(self):
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0

        # Tracks whether a pinch is currently "held", so we only fire a
        # click once per pinch (on the transition from released -> pinched),
        # not repeatedly every frame the fingers stay together.
        self.left_pinch_active = False
        self.right_pinch_active = False

        # Vertical position of the hand from the previous frame, used to
        # compute movement deltas while scrolling.
        self.previous_scroll_y = None

    @staticmethod
    def _finger_is_raised(landmarks, tip_index, pip_index):
        """
        A finger is considered "raised" if its tip is above (smaller y in
        image coordinates) its own pip joint by a small margin. Works for
        index/middle since they point roughly upward when extended.
        """
        return landmarks[tip_index][1] < landmarks[pip_index][1] - 10

    def detect_left_click(self, landmarks):
        """Thumb-tip <-> index-tip pinch => left click (with cooldown)."""
        distance = euclidean_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        is_pinching = distance < LEFT_CLICK_THRESHOLD

        clicked = False
        now = time.time()

        if is_pinching and not self.left_pinch_active:
            # New pinch just started -> eligible for a click if cooldown passed.
            if now - self.last_left_click_time >= CLICK_COOLDOWN:
                clicked = True
                self.last_left_click_time = now

        # Track pinch state so the click only fires once per pinch gesture.
        self.left_pinch_active = is_pinching
        return clicked

    def detect_right_click(self, landmarks):
        """Thumb-tip <-> middle-tip pinch => right click (with cooldown)."""
        distance = euclidean_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_TIP])
        is_pinching = distance < RIGHT_CLICK_THRESHOLD

        clicked = False
        now = time.time()

        if is_pinching and not self.right_pinch_active:
            if now - self.last_right_click_time >= CLICK_COOLDOWN:
                clicked = True
                self.last_right_click_time = now

        self.right_pinch_active = is_pinching
        return clicked

    def is_scroll_gesture(self, landmarks):
        """Scroll mode = index AND middle finger both raised."""
        index_up = self._finger_is_raised(landmarks, INDEX_TIP, INDEX_PIP)
        middle_up = self._finger_is_raised(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        return index_up and middle_up

    def handle_scroll(self, landmarks):
        """
        Computes vertical hand movement between frames and issues a scroll
        command if the movement exceeds the dead-zone threshold. Moving
        the hand UP scrolls UP (positive scroll), DOWN scrolls DOWN.
        """
        # Use the midpoint between index and middle tips as the tracked
        # reference point for scroll movement.
        current_y = (landmarks[INDEX_TIP][1] + landmarks[MIDDLE_TIP][1]) / 2

        if self.previous_scroll_y is None:
            self.previous_scroll_y = current_y
            return

        delta_y = self.previous_scroll_y - current_y  # positive = moved up

        if abs(delta_y) >= SCROLL_THRESHOLD:
            # Normalize direction to a single scroll "tick" per qualifying
            # frame so speed doesn't spike on very fast motions.
            direction = 1 if delta_y > 0 else -1
            pyautogui.scroll(direction * SCROLL_SPEED)
            self.previous_scroll_y = current_y
        # If movement is below threshold, don't update previous_scroll_y —
        # this avoids slowly "drifting" the reference point from noise.

    def reset_scroll_reference(self):
        """Call when leaving scroll mode so the next entry starts fresh."""
        self.previous_scroll_y = None


class CursorController:
    """Maps a webcam-space point to a smoothed, clamped screen coordinate."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Usable region inside the margin, in webcam pixel coordinates.
        margin_x = int(frame_width * FRAME_MARGIN)
        margin_y = int(frame_height * FRAME_MARGIN)
        self.min_x, self.max_x = margin_x, frame_width - margin_x
        self.min_y, self.max_y = margin_y, frame_height - margin_y

        self.smooth_x = SCREEN_WIDTH / 2
        self.smooth_y = SCREEN_HEIGHT / 2

    def map_to_screen(self, finger_x, finger_y):
        """
        Converts a fingertip pixel position (already mirrored, so left/right
        motion matches the user intuitively) into a screen coordinate,
        applying the margin so the edges of the usable area reach the
        full 0..screen_size range.
        """
        # Clamp into the usable (post-margin) region first.
        clamped_x = clamp(finger_x, self.min_x, self.max_x)
        clamped_y = clamp(finger_y, self.min_y, self.max_y)

        # Normalize 0..1 across the usable region.
        norm_x = (clamped_x - self.min_x) / (self.max_x - self.min_x)
        norm_y = (clamped_y - self.min_y) / (self.max_y - self.min_y)

        # Scale to screen dimensions.
        target_x = norm_x * SCREEN_WIDTH
        target_y = norm_y * SCREEN_HEIGHT
        return target_x, target_y

    def move_to(self, finger_x, finger_y):
        """Smooths toward the target screen point and moves the OS cursor."""
        target_x, target_y = self.map_to_screen(finger_x, finger_y)

        # Exponential smoothing (see SMOOTHING_FACTOR comment above).
        self.smooth_x += (target_x - self.smooth_x) * SMOOTHING_FACTOR
        self.smooth_y += (target_y - self.smooth_y) * SMOOTHING_FACTOR

        # Final safety clamp — coordinates must always be valid on-screen.
        final_x = int(clamp(self.smooth_x, 0, SCREEN_WIDTH - 1))
        final_y = int(clamp(self.smooth_y, 0, SCREEN_HEIGHT - 1))

        pyautogui.moveTo(final_x, final_y)

    def draw_usable_area(self, frame):
        """Draws the margin boundary on the frame for visual reference."""
        cv2.rectangle(
            frame,
            (self.min_x, self.min_y),
            (self.max_x, self.max_y),
            (0, 255, 0),
            2,
        )


def draw_status_text(frame, text, position, color=(0, 255, 255)):
    """Small helper to keep on-screen text drawing calls tidy."""
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened():
        print("Error: could not open webcam. Check camera index/permissions.")
        return

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    tracker = HandTracker()
    recognizer = GestureRecognizer()
    cursor = CursorController(frame_width, frame_height)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Warning: failed to read frame from webcam. Retrying...")
                continue

            # Mirror the feed so hand movement matches on-screen movement
            # (moving your hand right moves the view/cursor right).
            frame = cv2.flip(frame, 1)

            cursor.draw_usable_area(frame)

            landmarks = tracker.process(frame)

            if landmarks is not None:
                index_x, index_y = landmarks[INDEX_TIP]

                if recognizer.is_scroll_gesture(landmarks):
                    # Scroll mode: do NOT move the cursor or allow clicks
                    # while scrolling, to avoid conflicting actions.
                    recognizer.handle_scroll(landmarks)
                    draw_status_text(frame, "SCROLL MODE", (10, 30), (255, 200, 0))
                else:
                    recognizer.reset_scroll_reference()

                    # Move cursor using the index fingertip.
                    cursor.move_to(index_x, index_y)

                    # Check click gestures. Left/right are mutually exclusive
                    # per frame (checked independently, but a real pinch of
                    # one kind is geometrically unlikely to also satisfy the
                    # other at the same time given typical hand poses).
                    if recognizer.detect_left_click(landmarks):
                        pyautogui.click(button="left")
                        draw_status_text(frame, "LEFT CLICK", (10, 30), (0, 255, 0))
                    elif recognizer.detect_right_click(landmarks):
                        pyautogui.click(button="right")
                        draw_status_text(frame, "RIGHT CLICK", (10, 30), (0, 0, 255))

                cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), -1)
            else:
                # No hand detected — keep the app running, just skip
                # movement/gesture logic this frame.
                recognizer.reset_scroll_reference()
                draw_status_text(frame, "No hand detected", (10, 30), (0, 0, 255))

            cv2.imshow("Hand Gesture Virtual Mouse (press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Guaranteed cleanup even if an exception occurs mid-loop.
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
