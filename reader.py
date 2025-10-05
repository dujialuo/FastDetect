from utils import xp
import os
from Config import Config
class SlidingComplex64Reader: # TODO reader slow unbuffered
    dtype = xp.complex64
    itemsize = 8  # complex64

    def __init__(self, file_path: str, tstart: int):
        """
        Initializes the reader for a complex64 binary file.

        Args:
            file_path (str): The path to the binary file.
        """
        self.file_path = file_path
        self.tstart = tstart  # start time in samples (int)
        file_size = os.path.getsize(file_path)
        complex64_size = xp.dtype(xp.complex64).itemsize
        assert complex64_size == 8
        print(f"{file_path=} Size in Number of symbols: {file_size // complex64_size // Config.nsamp}")

    def get(self, start: int, end: int):
        """
        Reads a chunk of data from the file.

        Args:
            start (int): The starting index (in terms of complex64 elements).
            length (int): The number of complex64 elements to read.

        Returns:
            xp.ndarray: An array containing the requested data.
        """
        assert end > start
        byte_start = (start + self.tstart) * self.itemsize
        byte_length = (end - start) * self.itemsize

        try:
            with open(self.file_path, 'rb') as f:
                f.seek(byte_start)
                data_bytes = f.read(byte_length)

            if len(data_bytes) != byte_length:
                raise IOError(f"Read fewer bytes than expected. Expected {byte_length}, got {len(data_bytes)}")

            # Convert the bytes to a NumPy/CuPy array
            # Note: The .frombuffer method is efficient as it creates a view of the bytes.
            # We then copy it to the appropriate device (CPU or GPU) if necessary.
            data_array = xp.frombuffer(data_bytes, dtype=self.dtype)
            return data_array.astype(xp.complex128)

        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None