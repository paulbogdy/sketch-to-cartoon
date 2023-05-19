import os
import socket
import io
import struct
from typing import List, Tuple
from PIL import Image
import time


class StrategyServer:
    def __init__(self, socket_name: str = "strategy_server.socket"):
        self.socket_file = os.path.join("/tmp", socket_name)
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Remove the socket file if it already exists
        try:
            os.unlink(self.socket_file)
        except OSError:
            if os.path.exists(self.socket_file):
                raise

        # Bind the socket to the file path
        self.server_socket.bind(self.socket_file)

    def run_server(self) -> None:
        # Start listening for connections
        self.server_socket.listen(1)
        print("Listening for connections...")

        while True:
            self.connection, _ = self.server_socket.accept()

            print("Connection established")
            try:
                task_type_data = self.connection.recv(1)
                task_type = struct.unpack('!B', task_type_data)[0]

                sketch = self.receive_sketch()

                if task_type == 0:
                    print("Received task: generate images")
                    num_samples = self.receive_num_samples()
                    generated_images = self.generate_images(sketch, num_samples)
                    self.send_images(generated_images)
                else:
                    print("Received task: generate shadow")
                    num_samples = self.receive_num_samples()
                    generated_shadow = self.generate_shadow(sketch, num_samples)
                    print("Sending generated shadow...", end=' ')
                    self.send_one_imag(generated_shadow)
                    print("Done")

            finally:
                # Close the connection
                self.connection.close()

    def receive_sketch(self) -> Image.Image:
        print("Receiving sketch...", end=' ')
        sketch_size_data = self.connection.recv(4)
        sketch_size = struct.unpack('!I', sketch_size_data)[0]

        sketch_data = b''
        while len(sketch_data) < sketch_size:
            chunk = self.connection.recv(min(sketch_size - len(sketch_data), 4096))
            if not chunk:
                break
            sketch_data += chunk

        img = Image.open(io.BytesIO(sketch_data)).convert('L')
        print("Done")
        return img

    def receive_num_samples(self) -> int:
        print("Receiving number of samples...", end=' ')
        num_samples_data = self.connection.recv(4)
        num_samples = struct.unpack('!I', num_samples_data)[0]
        print(f"Done -> {num_samples} samples")
        return num_samples

    def generate_images(self, sketch: Image.Image, num_samples: int) -> List[Image.Image]:
        raise NotImplementedError

    def send_images(self, images: List[Image.Image]) -> None:
        print("Sending generated images...", end=' ')
        for img in images:
            self.send_one_imag(img)
        print("Done")

    def generate_shadow(self, sketch: Image.Image, num_samples: int) -> Image.Image:
        raise NotImplementedError

    def send_one_imag(self, img: Image) -> None:
        # Convert the image to PNG format and get its byte data
        img_byte_data = io.BytesIO()
        img.save(img_byte_data, format='PNG')
        img_byte_data = img_byte_data.getvalue()

        # Send the image size and data
        self.connection.sendall(struct.pack('!I', len(img_byte_data)))
        self.connection.sendall(img_byte_data)

