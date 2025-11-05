# Import Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import Kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import Kivy utilities
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import OpenCV, TensorFlow, and other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
from layers import L1Dist

# Define CamApp class
class CamApp(App):

    def build(self):
        print("[INFO] App is starting...")  # Debugging message

        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load the TensorFlow/Keras model inside build()
        try:
            print("[INFO] Loading Siamese Model...")
            self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist': L1Dist})
            print("[INFO] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            exit()

        # Setup OpenCV video capture
        print("[INFO] Initializing Camera...")
        self.capture = cv2.VideoCapture(0)  # Change to 1 or 2 if needed

        if not self.capture.isOpened():
            print("[ERROR] Could not open webcam. Exiting...")
            exit()

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS

        return layout

    # Continuously update webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            print("[ERROR] Failed to capture image from camera.")
            return

        frame = frame[120:370, 200:450]  # Crop the region of interest

        # Convert OpenCV frame to Kivy texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocess image for model input
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)

        # Resize and normalize image
        img = tf.image.resize(img, (100, 100)) / 255.0
        return img

    # Verify face using Siamese Network
    def verify(self, *args):
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Capture input image
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        if not ret:
            print("[ERROR] Failed to capture verification image.")
            return

        frame = frame[120:370, 200:450]  # Crop image
        cv2.imwrite(SAVE_PATH, frame)

        # Compare input image with verification images
        results = []
        verification_images_path = os.path.join('application_data', 'verification_images')

        for image in os.listdir(verification_images_path):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join(verification_images_path, image))

            # Ensure correct input format for model
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        # Calculate detection & verification scores
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(verification_images_path))
        verified = verification > verification_threshold

        # Display result
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log details
        Logger.info(f"Results: {results}")
        Logger.info(f"Detection Count: {detection}")
        Logger.info(f"Verification Score: {verification}")
        Logger.info(f"Final Decision: {verified}")

        return results, verified


if __name__ == '__main__':
    CamApp().run()
