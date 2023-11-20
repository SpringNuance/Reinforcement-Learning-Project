"""Wrapper for recording videos."""
import os
from typing import Callable, Optional

import gym
from gym import logger
"""A wrapper for video recording environments by rolling it out, frame by frame."""
import json
import os
import os.path
import tempfile
from typing import List, Optional

from gym import error, logger


class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame.

    It comes with an ``enabled`` option, so you can still use the same code on episodes where you don't want to record video.

    Note:
        You are responsible for calling :meth:`close` on a created VideoRecorder, or else you may leak an encoder process.
    """

    def __init__(
        self,
        env,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        try:
            # check that moviepy is now installed
            import moviepy  # noqa: F401
        except ImportError:
            raise error.DependencyNotInstalled(
                "MoviePy is not installed, run `pip install moviepy`"
            )

        self._async = env.metadata.get("semantics.async")
        self.enabled = enabled
        self._closed = False

        self.render_history = []
        self.env = env

        self.render_mode = env.render_mode

        if "rgb_array_list" != self.render_mode and "rgb_array" != self.render_mode:
            logger.warn(
                f"Disabling video recorder because environment {env} was not initialized with any compatible video "
                "mode between `rgb_array` and `rgb_array_list`"
            )
            # Disable since the environment has not been initialized with a compatible `render_mode`
            self.enabled = False

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}."
            )

        self.frames_per_sec = env.metadata.get("render_fps", 30)

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        logger.info(f"Starting new video recorder writing to {self.path}")
        self.recorded_frames = []

    @property
    def functional(self):
        """Returns if the video recorder is functional, is enabled and not broken."""
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        frames = self.env.render()

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frames is None:
            if self._async:
                return
            else:
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        else:
            if not isinstance(frames, list):
                frames = [frames]
            self.recorded_frames.extend(frames)


    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        # First close the environment
        self.env.close()

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            logger.debug(f"Closing video encoder: path={self.path}")
            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.path)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True

    def write_metadata(self):
        """Writes metadata to metadata path."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __del__(self):
        """Closes the environment correctly when the recorder is deleted."""
        # Make sure we've closed up shop when garbage collecting
        self.close()


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordVideo(gym.Wrapper):
    """This wrapper records videos of rollouts.

    Usually, you only want to record episodes intermittently, say every hundredth episode.
    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `terminated` or `truncated` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
        """
        super().__init__(env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[VideoRecorder] = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            if self.recording:
                return self.video_recorder.last_frame
            else:
                return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()

    def __del__(self):
        """Closes the video recorder."""
        self.close_video_recorder()