from pathlib import Path
from typing import Any, Iterator, Union

import cv2


class VideoCapture:
    """
    Base class for video capture.
    Provides common properties and an iterator interface for video capture streams.
    Attributes:
        capture (cv2.VideoCapture): The underlying video capture object.
    """
    def __init__(self, source: str) -> None:
        """
        Initialize the Capture instance.
        Args:
            capture (cv2.VideoCapture): The OpenCV video capture object.
        """
        self.capture = cv2.VideoCapture(source)

    @property
    def fps(self) -> float:
        """Return the frames per second of the capture."""
        return self.capture.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Return the width of the capture frames."""
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Return the height of the capture frames."""
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[Any]:
        """
        Return an iterator over frames of the capture.

        Raises:
            IOError: If the capture device is not opened.
        """
        if self.capture.isOpened():
            return self
        raise IOError("Capture device is not opened.")

    def __next__(self) -> Any:
        """
        Return the next frame from the capture.

        Raises:
            StopIteration: If no more frames are available.
        """
        ret, frame = self.capture.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame

    def release(self) -> None:
        """
        Release the underlying video capture resources.
        """
        if self.capture.isOpened():
            self.capture.release()


class VideoReader(VideoCapture):
    """
    A class for reading video files using OpenCV.

    Extends the Capture class to provide file-specific functionalities such as
    retrieving frame count and random access to frames.

    Attributes:
        source (Path): Path to the video file.
    """

    def __init__(self, filename: Union[str, Path]) -> None:
        """
        Initialize a VideoReader instance.

        Args:
            filename (str | Path): Path to the video file.

        Raises:
            AssertionError: If the file does not exist.
            IOError: If the video file cannot be opened.
        """
        source = Path(filename).resolve()
        assert source.exists(), f"The input file {source} does not exist."
        super().__init__(source.as_posix())
        if not self.capture.isOpened():
            raise IOError(f"Failed to open video file {source}")

    @property
    def frame_count(self) -> int:
        """Return the total number of frames in the video."""
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def codec(self) -> int:
        """Return the codec of the video file."""
        return int(self.capture.get(cv2.CAP_PROP_FOURCC))

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve a specific frame by index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            The frame image if available; otherwise, None.
        """
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.capture.read()
        return frame if ret else None

    def __len__(self) -> int:
        """
        Return the total number of frames in the video.

        Returns:
            The number of frames.
        """
        return self.frame_count

    def __repr__(self) -> str:
        """
        Return a string representation of the VideoReader instance.

        Returns:
            A string containing the file name and key video properties.
        """
        return (
            f"VideoReader(filename={self.source}, fps={self.fps}, "
            f"total_frames={self.frame_count}, width={self.width}, height={self.height})"
        )


class LiveVideoReader(VideoCapture):
    """
    A class for reading live video streams using OpenCV.

    Extends the Capture class to handle live streams. Note that frame counting
    and random access are not applicable for live streams.
    """

    def __init__(self, stream: str) -> None:
        """
        Initialize a LiveVideoReader instance.

        Args:
            stream (str): The URL or device index of the live stream.

        Raises:
            IOError: If the live stream cannot be opened.
        """
        super().__init__(stream)

    @property
    def frame_count(self) -> int:
        """
        Frame count is not available for live streams.

        Raises:
            AttributeError: Always, because frame count is not supported.
        """
        raise AttributeError("Frame count is not available for live streams.")

    def __repr__(self) -> str:
        """
        Return a string representation of the LiveVideoReader instance.

        Returns:
            A string containing key live stream properties.
        """
        return f"LiveVideoReader(fps={self.fps}, width={self.width}, height={self.height})"
