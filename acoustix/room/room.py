"""
Abstract implementation of a shoebox room.
Both backend are supported:
- PyRoomAcoustics (https://github.com/LCAV/pyroomacoustics) implements the Image Source Model and
    a ray tracing algorithm.
- GpuRIR (https://github.com/DavidDiazGuerra/gpuRIR) provides a cuda implementation of the Image
    Source Model.
"""

import logging
import math
from collections import OrderedDict
from enum import IntEnum
from itertools import product
from pathlib import Path
from typing import Iterator, Optional

import exputils as eu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from .audio_objects import AudioObject, Microphone, Source

#########################
# Discrete orientations #
#########################


# Note: the order matters as we want to rotate by an angle of +90° when incrementing the index by 1.
class Orientation(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


NORMAL_VECTOR: dict[Orientation, np.ndarray] = {
    Orientation.UP: np.array([0, -1, 0]),
    Orientation.DOWN: np.array([0, 1, 0]),
    Orientation.LEFT: np.array([-1, 0, 0]),
    Orientation.RIGHT: np.array([1, 0, 0]),
}


def orientation_from_vec(vec: np.ndarray) -> Orientation:
    """
    Convert a 2D/3D vector to a discrete Orientation enum.

    Args:
        vec: 2D or 3D orientation vector

    Returns:
        Corresponding Orientation enum value
    """
    assert vec.ndim == 1
    assert len(vec) in (2, 3)

    if vec[0] == 0:
        if vec[1] == -1:
            return Orientation.UP

        else:
            return Orientation.DOWN

    else:
        if vec[0] == -1:
            return Orientation.LEFT

        else:
            return Orientation.RIGHT


class IllegalPosition(Exception):
    pass


class Room:
    """
    Abstract class representing a room.

    Attributes:
        size_x (float):                             The room width (in meters).
        size_y (float):                             The room height (in meters).
        height (float):                             Height of the room (in meters).
        rt_60 (float):                              Desired reverberation time: time for the RIR to
                                                        reach 60dB of attenuation (in seconds).
        max_time (float):                           Length of a RIR (in seconds).
        freq_sampling (int):                        The sampling frequency used in the simulator.
        delta_x (float):                            The distance between two microphones from the
                                                        grid (in the x direction).
        delta_y (float):                            The distance between two microphones from the
                                                        grid (in the y direction).
        n_x (int):                                  The width of the grid (number of microphones
                                                        per row).
        n_y (int):                                  The height of the grid (number of microphones
                                                        per column).
        delta_theta (float):                        Angular sampling resolution (in radians).
        grid_rir (np.ndarray):                      The precomputed RIR at the grid positions.
        sources (OrderedDict[str, Source]):         The audio sources in the room.
        microphones (OrderedDict[str, Microphone]): The microphones in the room.
    """

    default_config: eu.AttrDict = eu.AttrDict(
        size_x=7,
        size_y=4,
        height=3,
        rt_60=0.3,
        max_time=-1,
        sampling_frequency=16_000,
        grid=eu.AttrDict(
            delta_x=0.5,
            delta_y=0.5,
            height=1.5,
            add_grid_mics=False,
        ),
    )

    def __init__(
        self,
        config: eu.AttrDict = None,
        **kwargs,
    ) -> None:
        """
        Init function for a Room object.

        Args:
            size_x (float):             The room width (in meters).
            size_y (float):             The room height (in meters).
            height (float):             Height of the room (in meters).
            rt_60 (float):              Desired reverberation time: time for the RIR to reach 60dB
                                            of attenuation (in seconds).
            max_time (float):           Length of a RIR (in seconds).
            sampling_frequency (int):   The sampling frequency used in the simulator.
        """
        self.config: eu.AttrDict = eu.combine_dicts(
            kwargs,
            config,
            self.default_config,
        )

        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.size_x: float = self.config.size_x
        self.size_y: float = self.config.size_y
        self.height = self.config.height

        # Audio settings
        self.max_time: float = self.config.max_time
        self._max_time_auto: bool = self.max_time < 0
        self.sampling_frequency: float = self.config.sampling_frequency

        # Spatial sampling
        self.delta_x: float | None = None
        self.delta_y: float | None = None

        self.grid: np.ndarray

        # precomputed RIR
        self.rir: np.ndarray

        # Reverberation time (in s)
        self.rt_60: float = self.config.rt_60

        # If the precomputed RIRs are up to date
        self.rir_up_to_date: bool = False

        # If the audio simulation (i.e. the microphones listened signals) are up to date
        self.simulation_up_to_date: bool = False

        # Sources
        self.sources: OrderedDict[str, Source] = OrderedDict()

        # Microphones
        self.microphones: OrderedDict[str, Microphone] = OrderedDict()

    ########
    # GRID #
    ########

    @property
    def grid_x_coords(self) -> np.ndarray:
        return np.arange(
            start=self.delta_x / 2,
            stop=self.size_x,
            step=self.delta_x,
        )

    @property
    def grid_y_coords(self) -> np.ndarray:
        return np.arange(
            start=self.delta_y / 2,
            stop=self.size_y,
            step=self.delta_y,
        )

    def _init_simulation_params(self) -> None:
        raise NotImplementedError

    def set_rt60(self, rt_60: float) -> None:
        self.rir_up_to_date = False
        self.simulation_up_to_date = False

        self._logger.warning("UPDATING RT_60 from %f to %f", self.rt_60, rt_60)

        self.rt_60 = rt_60

        self._init_simulation_params()

    @property
    def n_x(self) -> int:
        return math.ceil((self.size_x / self.delta_x) - 0.5)

    @property
    def n_y(self) -> int:
        return math.ceil((self.size_y / self.delta_y) - 0.5)

    def init_grid(
        self,
        add_grid_mics: bool = False,
        mic_pattern: str = "omni",
        config: eu.AttrDict = None,
        **kwargs,
    ) -> None:
        """
        Initialize a grid of evenly spaced microphones.

        Args:
            delta_x (float):        Spatial resolution in the x direction. (default: 0.5m)
            delta_y (float):        Spatial resolution in the x direction. (default: 0.5m)
            delta_theta (float):    Angular resolution. (default: pi/4)
            height (float):         The height of the microphone array. (default: 1.5m)
        """
        self.config.grid = eu.combine_dicts(
            self.config.grid,
            config,
            kwargs,
        )

        if hasattr(self, "grid") and self.grid is not None:
            self._logger.debug("This room's grid has already been initialized")
            return

        # Check that the height is in the room.
        assert 0 < self.config.grid.height < self.height

        # Spatial sampling
        self.delta_x = self.config.grid.delta_x
        self.delta_y = self.config.grid.delta_y

        # Store the grid coordinates
        self.grid = np.zeros(shape=(self.n_x, self.n_y, 2))

        for x_grid, mic_x in enumerate(self.grid_x_coords):
            for y_grid, mic_y in enumerate(self.grid_y_coords):
                self.grid[x_grid, y_grid, 0] = mic_x
                self.grid[x_grid, y_grid, 1] = mic_y

                if add_grid_mics:
                    self.add_microphone(
                        mic=Microphone(
                            name=f"grid_{mic_x}_{mic_y}",
                            position=np.array(
                                [
                                    mic_x,
                                    mic_y,
                                    self.config.grid.height,
                                ],
                            ),
                            pattern=mic_pattern,
                        )
                    )

        assert x_grid == self.n_x - 1 and y_grid == self.n_y - 1

        if add_grid_mics:
            assert len(self.microphones) == self.n_x * self.n_y

    def grid_position_iterator(self) -> Iterator[tuple[float, float]]:
        """
        Iterate over the 2d positions from the spatial grid.

        Returns:
            Iterator[tuple[float, float]]:  An Iterator of 2d coordinates of the grid points.
        """
        for x_coord in self.grid_x_coords:
            for y_coord in self.grid_y_coords:
                yield float(x_coord), float(y_coord)

    def get_room_dims(self) -> np.ndarray:
        """
        Get the room dimensions in a numpy array.

        Returns:
            room_dims (np.ndarray): The dimensions of the room. shape: (3,)
        """
        return np.array(
            [
                self.size_x,
                self.size_y,
                self.height,
            ],
        )

    def is_in_room(self, position: np.ndarray) -> bool:
        """
        Checks if a point is inside the room.

        Args:
            location (np.ndarray):  The coordinates of a 2D or 3D point.

        Returns:
            is_in_room (bool):  True if and only if the point is inside the room.
        """
        return bool((position > 0).all() and (position < self.get_room_dims()).all())

    def is_circle_in_room(self, center_position: np.ndarray, radius: float) -> bool:
        x: float = center_position[0]
        y: float = center_position[1]

        # LEFT
        if x - radius < 0:
            return False
        # RIGHT
        if x + radius > self.get_room_dims()[0]:
            return False
        # TOP
        if y - radius < 0:
            return False
        # BOTTOM
        if y + radius > self.get_room_dims()[1]:
            return False
        return True

    def assert_is_in_room(self, position: np.ndarray) -> None:
        if not self.is_in_room(position=position):
            raise IllegalPosition(
                f"Position {position} is out of bound for this room of dimensions:\n"
                f"\tsize_x={self.size_x}\n\tsize_y={self.size_y}\n\theight={self.height}"
            )

    def _get_audio_object_dict(
        self,
        audio_object: AudioObject,
    ) -> OrderedDict[str, AudioObject]:
        """
        Get the dict of AudioObject (attribute) corresponding to the given AudioObject.
        In practice,
        - if given a Microphone, return `self.microphones`
        - if given a Source, return `self.sources`

        Args:
            audio_object (AudioObject): Either a Microphone or a Source object.

        Returns:
            dest_dict (OrderedDict[str, AudioObject]):  The corresponding dict.
        """
        dest_dict: OrderedDict[str, AudioObject]
        if isinstance(audio_object, Source):
            dest_dict = self.sources  # type: ignore
        elif isinstance(audio_object, Microphone):
            dest_dict = self.microphones  # type: ignore
        else:
            raise Exception(f"Unknown type: {type(AudioObject)}")

        return dest_dict

    def _add(
        self,
        audio_object: AudioObject,
        clip_to_grid: bool,
    ) -> None:
        """
        Private methode to add an AudioObject to its corresponding object.

        Args:
            audio_object (AudioObject):     An audio object.
            clip_to_grid (bool):            If True, move the audio_object to the closest grid
                                                point.
        """
        dest_dict: OrderedDict[str, AudioObject] = self._get_audio_object_dict(
            audio_object=audio_object
        )

        assert audio_object.name not in dest_dict, (
            f"{audio_object.name} already exists in the room."
        )

        if clip_to_grid:
            loc: np.ndarray = audio_object.position
            n_x: int = round((loc[0] / self.delta_x) - 0.5)
            n_y: int = round((loc[1] / self.delta_y) - 0.5)
            clipped_x: float = (n_x + 0.5) * self.delta_x
            clipped_y: float = (n_y + 0.5) * self.delta_y
            audio_object.set_position(
                np.array(
                    [
                        clipped_x,
                        clipped_y,
                        loc[2],
                    ],
                ),
            )

        assert self.is_in_room(
            position=audio_object.position,
        ), (
            f"Audio object {audio_object.name} has invalid position ({audio_object.position})"
            f" | Room has dimensions {self.get_room_dims()}"
        )

        audio_object.index = len(dest_dict)

        dest_dict[audio_object.name] = audio_object

        self.rir_up_to_date = False
        self.simulation_up_to_date = False

    def add_microphone(
        self,
        mic: Microphone,
        clip_to_grid: bool = False,
    ) -> None:
        """
        Add a microphone to the room.

        Args:
            mic (Microphone):       A Microphone object.
            clip_to_grid (bool):    If True, move the microphone to the closest grid point.
        """
        self._add(
            audio_object=mic,
            clip_to_grid=clip_to_grid,
        )

    def add_microphones(
        self,
        microphones_list: list[Microphone],
        clip_to_grid: bool = False,
    ) -> None:
        """
        Add several microphones to the list.

        Args:
            microphones_list (list[Microphone]):    A list of Microphone objects.
            clip_to_grid (bool):                    If True, move the microphone to the closest grid
                                                        point.
        """
        for mic in microphones_list:
            self.add_microphone(
                mic=mic,
                clip_to_grid=clip_to_grid,
            )

    def add_source(
        self,
        source: Source,
        clip_to_grid: bool = False,
    ) -> None:
        """
        Add an audio source to the room.

        Args:
            source (Source):        An audio source.
            clip_to_grid (bool):    If True, move the source to the closest grid point.
        """
        self._add(
            audio_object=source,
            clip_to_grid=clip_to_grid,
        )

    def _move(
        self,
        audio_object_name: str,
        audio_object_dict: OrderedDict[str, AudioObject],
        new_position: Optional[np.ndarray] = None,
        new_orientation: Optional[np.ndarray] = None,
        delta: Optional[np.ndarray] = None,
    ) -> None:
        """
        Move an audio object to a new location.

        Args:
            audio_object_name (str):                            The name of the object to move.
            new_position (np.ndarray):                          The new position of the object.
            new_orientation (np.ndarray):                       The new orientation of the object.
            audio_object_dict (OrderedDict[str, AudioObject]):  The dictionnary containing the,
                                                                    AudioObject to move.
            delta (np.ndarray):                                 A displacement vector can be
                                                                    provided instead of an absolute
                                                                    new location:
                                                                    new_location = current_location
                                                                        + delta.
                                                                    Shape=(2-3,)
        """
        assert audio_object_name in audio_object_dict, (
            f"Audio object '{audio_object_name}' is not in the room.\n"
            f"Present objects: {list(audio_object_dict.keys())}"
        )
        assert new_position is None or delta is None

        audio_object: AudioObject = audio_object_dict[audio_object_name]
        if delta is not None:
            if len(delta) == 2:
                delta = np.append(delta, 0)
            assert delta is not None
            new_position = audio_object.position + delta

        assert new_position is not None
        if len(new_position) == 2:
            new_position = np.append(
                new_position,
                audio_object.position[2],
            )

        assert new_position is not None
        self.assert_is_in_room(position=new_position)

        # self.logger.debug("old location: %s", audio_object.location)
        audio_object.set_position(pos=new_position)
        # self.logger.debug("new location: %s", audio_object.location)

        if new_orientation is not None:
            audio_object.set_orientation(orientation=new_orientation)

        self.rir_up_to_date = False
        self.simulation_up_to_date = False

    def move_source(
        self,
        source_name: str,
        new_position: Optional[np.ndarray] = None,
        new_orientation: Optional[np.ndarray] = None,
        delta: Optional[np.ndarray] = None,
    ) -> None:
        """
        Move an audio source to a new location.

        Args:
            source_name (str):              The name of the source to move.
            new_position: (np.ndarray):      The new location. Shape=(2-3,)
            new_orientation (np.ndarray):   The new orientation of the object.
            delta (np.ndarray):             A displacement vector can be provided instead of an
                                                absolute new location:
                                                new_location = current_location + delta.
                                                shape=(2-3,)
        """
        self._move(
            audio_object_name=source_name,
            new_position=new_position,
            new_orientation=new_orientation,
            delta=delta,
            audio_object_dict=self.sources,  # type: ignore
        )

    def move_microphone(
        self,
        mic_name: str,
        new_position: Optional[np.ndarray] = None,
        new_orientation: Optional[np.ndarray] = None,
        delta: Optional[np.ndarray] = None,
    ) -> None:
        """
        Move a microphone to a new location.

        Args:
            mic_name (str):                 The name of the microphone to move.
            new_position (np.ndarray):      The new position. Shape=(2-3,)
            new_orientation (np.ndarray):   The new orientation of the object.
            delta (np.ndarray):             A displacement vector can be provided instead of an
                                                absolute new position:
                                                new_position = current_position + delta.
                                                shape=(2-3,)
        """
        self._move(
            audio_object_name=mic_name,
            new_position=new_position,
            new_orientation=new_orientation,
            delta=delta,
            audio_object_dict=self.microphones,  # type: ignore
        )

    def _remove(
        self,
        audio_object_name: str,
        audio_object_dict: OrderedDict[str, AudioObject],
    ) -> None:
        """
        Private method to remove a source or a microphone.

        Args:
            audio_object_name (str):                            The name of the AudioObject to
                                                                    remove.
            audio_object_dict (OrderedDict[str, AudioObject]):  The dictionnary containing the
                                                                    AudioObject to remove.
        """
        if audio_object_name in audio_object_dict:
            removed_audio_object = audio_object_dict.pop(audio_object_name)

            # Update indices
            for object_index, audio_object in enumerate(audio_object_dict.values()):
                if object_index > removed_audio_object.index:
                    audio_object.index -= 1

            self.rir_up_to_date = False
            self.simulation_up_to_date = False
        else:
            self._logger.error("'%s' does not exist", audio_object_name)

    def get_source_index(self, source_name: str) -> int:
        """
        Get the index of an audio source.

        Args:
            source_name (str):  A source name.
        """
        assert source_name in self.sources
        return self.sources[source_name].index

    def get_source_from_index(self, index: int) -> Source:
        """
        Get the source object corresponding to the given index.

        Args:
            index (int):        The index of the source.

        Returns:
            source (Source):    The corresponding source.
        """
        return list(self.sources.values())[index]

    def get_mic_index(self, mic_name: str) -> int:
        """
        Get the index of a microphone.

        Args:
            mic_name (str):  A microphone name.
        """
        assert mic_name in self.microphones
        return self.microphones[mic_name].index

    def get_mic_from_index(self, index: int = -1) -> Microphone:
        """
        Get the microphone object corresponding to the given index.

        Args:
            index (int):    The index of the microphone.

        Returns:
            mic (Microphone):   The corresponding microphone.
        """
        return list(self.microphones.values())[index]

    def _get_position(
        self,
        audio_object_name: str,
        audio_object_dict: dict[str, AudioObject],
    ) -> np.ndarray:
        return audio_object_dict[audio_object_name].position

    def get_mic_position(self, mic_name: str) -> np.ndarray:
        return self._get_position(
            audio_object_name=mic_name,
            audio_object_dict=self.microphones,  # type: ignore
        )

    def get_source_position(self, source_name: str) -> np.ndarray:
        return self._get_position(
            audio_object_name=source_name,
            audio_object_dict=self.sources,  # type: ignore
        )

    def set_source_input_audio_signal(
        self,
        input_audio_signal: np.ndarray,
        delay: float = 0,
        source_name: str = "",
        source_index: int = -1,
    ) -> None:
        """
        Set the audio input signal for a given source.

        Args:
            input_audio_signal (np.ndarray):    The mono-aural signal (1D numpy array).
            delay (float):                      The delay before the audio starts.
            source_name (str):                  The name of the source.
            source_index (int):                 Alternatively it is possible to directly provide the
                                                    index of the source.
        """
        assert source_name != "" or source_index >= 0

        assert input_audio_signal.ndim == 1

        source: Source
        if source_name != "":
            assert source_name in self.sources, (
                f"Source '{source_name}' is not in the sources list: {self.sources}"
            )
            source = self.sources[source_name]
        else:
            source = self.get_source_from_index(index=source_index)

        source.signal = input_audio_signal
        source.delay = delay

        self.simulation_up_to_date = False

    def remove_microphone(self, mic_name: str) -> None:
        """
        Remove a microphone from the room.

        Args:
            mic_name (str): A microphone name.
        """
        self._remove(
            audio_object_name=mic_name,
            audio_object_dict=self.microphones,  # type: ignore
        )

    def remove_source(self, source_name: str) -> None:
        """
        Remove an audio source from the room.

        Args:
            mic_name (str): A source name.
        """
        self._remove(
            audio_object_name=source_name,
            audio_object_dict=self.sources,  # type: ignore
        )

    def get_random_grid_position(self) -> np.ndarray:
        """
        Generate a random 2D position sampled from the grid.

        Returns:
            grid_position (np.ndarray):     The sampled 2D position. shape=(2,)
        """
        x_coord: float = np.random.choice(
            np.arange(
                start=0 + self.delta_x / 2,
                stop=self.size_x,
                step=self.delta_x,
            )
        )
        y_coord: float = np.random.choice(
            np.arange(
                start=0 + self.delta_y / 2,
                stop=self.size_y,
                step=self.delta_y,
            )
        )

        return np.array([x_coord, y_coord])

    def get_random_position(
        self,
        height: float = -1,
        padding: float = 0.2,
    ) -> np.ndarray:
        """
        Get the coordinates of a randomly positionned 3d point.

        Args:
            height (float):     Provide a fixed height.
            padding (float):    Ensures the provided position is further than `padding` from the
                                    closest wall.

        Returns:
            position (np.ndarray):  The coordinates of the sampled point.
                                        shape: (3,)
        """
        x_coord: float = np.random.uniform(
            low=padding,
            high=self.size_x - padding,
        )
        y_coord: float = np.random.uniform(
            low=padding,
            high=self.size_y - padding,
        )

        z_coord: float
        if height >= 0:
            z_coord = height
        else:
            z_coord = np.random.random() * self.height

        return np.array(
            [
                x_coord,
                y_coord,
                z_coord,
            ],
        )

    def clear_sources(self) -> None:
        """
        Removes all the audio sources from the room.
        """
        while len(self.sources) > 0:
            self.sources.popitem()

        self.rir_up_to_date = False
        self.simulation_up_to_date = False

    def clear_microphones(self) -> None:
        """
        Removes all the microphones from the room.
        """
        while len(self.microphones) > 0:
            self.microphones.popitem()

    ##############
    # SIMULATION #
    ##############

    def pre_compute_rir(self) -> None:
        """
        Pre compute the RIR for every source-microphone pair.
        This method is implemented in inheriting classes.
        """
        raise NotImplementedError

    def _online_rir(
        self,
        point: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Compute the RIR for a given position directly.

        Args:
            point (np.ndarray): Coordinates of a 3D point representing the current location of the
                                    agent.
                                    shape=(2,)
            theta (float):      Orientation of the agent.


        Returns:
            rir (np.ndarray):   The Room Impulse Response at this location-orientation.
                                    shape = (self.max_time)
        """
        raise NotImplementedError

    def get_rir(
        self,
        source_name: str,
        mic_name: str,
    ) -> np.ndarray:
        """
        Get the Room Impulse Response function between a source and a microphone.

        Args:
            source_name (str):  The name of the source.
            mic_name (str):     The name of the microphone.

        Returns:
            rir (np.ndarray):   The RIR value (a 1D numpy array).
        """
        if not self.rir_up_to_date:
            self.pre_compute_rir()

        source_index: int = self.get_source_index(source_name=source_name)

        mic_index: int = self.get_mic_index(mic_name=mic_name)

        return self.rir[mic_index, source_index]

    def _compute_simulated_signal_length(
        self,
        max_rir_length: int,
    ) -> int:
        max_input_signal_length: int = max(
            [
                len(source.signal) + np.floor(source.delay * self.sampling_frequency)
                for source in self.sources.values()
                if source.is_active
            ]
        )
        simulated_signal_length: int = int(max_rir_length) + int(max_input_signal_length) - 1
        if simulated_signal_length % 2 == 1:
            simulated_signal_length += 1

        return simulated_signal_length

    def _compute_signal_at_mic(
        self,
        mic_index: int,
        simulated_signal_length: int,
    ) -> np.ndarray:
        """
        Compute the listened signal only at the provided microphone
        """
        mic_listened_signal: np.ndarray = np.zeros(
            shape=simulated_signal_length,
            dtype=np.float32,
        )

        for source_index, source in enumerate(self.sources.values()):
            if not source.is_active:
                continue
            source_signal: np.ndarray = source.signal
            if source_signal is None:
                continue

            # Get the precomputed rir
            rir: np.ndarray = self.rir[mic_index][source_index]
            starting_time_index: int = int(
                np.floor(
                    source.delay * self.sampling_frequency,
                ),
            )
            ending_time_index: int = starting_time_index + len(source_signal) + len(rir) - 1

            mic_listened_signal[starting_time_index:ending_time_index] += fftconvolve(
                in1=rir,
                in2=source_signal,
            )

        # Normalize the signal
        assert np.abs(mic_listened_signal).max() > 0
        mic_listened_signal /= np.abs(mic_listened_signal).max()

        return mic_listened_signal

    def simulate(
        self,
        force: bool = False,
    ) -> None:
        """
        Compute the audio signal listened by each microphone of the room.
        Code for this function comes from the PyRoomAcoustics library.
        """
        if (not force) and self.simulation_up_to_date:
            self._logger.info("Useless call to `simulate()`: the simulation is already up to date.")
            return

        if force or (not self.rir_up_to_date):
            self.pre_compute_rir()

        # self._logger.debug("Simulating audio")

        n_sources: int = len(self.sources)
        n_mics: int = len(self.microphones)

        max_rir_length: int = max(
            [len(self.rir[i][j]) for i, j in product(range(n_mics), range(n_sources))]
        )

        simulated_signal_length: int = self._compute_simulated_signal_length(
            max_rir_length=max_rir_length,
        )

        for mic_index, mic in enumerate(self.microphones.values()):
            mic.listened_signal = self._compute_signal_at_mic(
                mic_index=mic_index,
                simulated_signal_length=simulated_signal_length,
            )

        self.simulation_up_to_date = True

    def get_audio_at_mic(
        self,
        mic_name: str = "",
        mic_index: int = -1,
        save: bool = False,
        save_dir: str = "output/",
    ) -> np.ndarray:
        """
        Return the audio signal for a given microphone (using either its name, or directly its
        index).
        This function will start the simulation if needed.

        Args:
            mic_name (str):     A microphone name (optional).
            mic_index (int):    A microphone index (optional).

        Returns:
            target_audio (np.ndarray):  The audio signal at the requested position.
        """
        assert mic_name != "" or mic_index >= 0

        if not self.rir_up_to_date:
            self.pre_compute_rir()

        mic: Microphone
        if mic_name != "":
            mic = self.microphones[mic_name]
        else:
            mic = self.get_mic_from_index(index=mic_index)

        # Compute signal only for this mic
        if not self.simulation_up_to_date:
            n_sources: int = len(self.sources)

            max_rir_length: int = max(
                [len(self.rir[mic_index][source_index]) for source_index in range(n_sources)]
            )

            simulated_signal_length: int = self._compute_simulated_signal_length(
                max_rir_length=max_rir_length,
            )

            mic.listened_signal = self._compute_signal_at_mic(
                mic_index=mic_index,
                simulated_signal_length=simulated_signal_length,
            )

        if save:
            wavfile.write(
                filename=str(Path(save_dir) / (mic.name + ".wav")),
                rate=self.sampling_frequency,
                data=mic.listened_signal,
            )

        return mic.listened_signal

    def get_audio(
        self,
        save: bool = False,
        save_dir: str = "output/",
    ) -> np.ndarray:
        """
        Return the audio signal for all the microphones.
        This function will start the simulation if needed.

        Returns:
            listened_signals (np.ndarray):  The listened audio signal for each microphone.
                                                shape: (n_mics, T)
        """
        if not (self.rir_up_to_date and self.simulation_up_to_date):
            self.simulate()

        listened_signals: list[np.ndarray] = []
        for mic in self.microphones.values():
            if save:
                wavfile.write(
                    filename=str(
                        Path(save_dir) / (mic.name + ".wav"),
                    ),
                    rate=self.sampling_frequency,
                    data=mic.listened_signal,
                )
            listened_signals.append(mic.listened_signal)

        assert all(
            [
                len(listened_signal) == len(listened_signals[0])
                for listened_signal in listened_signals
            ]
        )

        return np.array(listened_signals)

    ########
    # MISC #
    ########

    def plot(
        self,
        show_grid: bool = True,
        show_sources: bool = True,
        show_mics: bool = True,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        self.figure: matplotlib.figure.Figure
        self.axes: matplotlib.axes.Axes
        self.figure, self.axes = plt.subplots()

        dpi: float = float(self.figure.get_dpi())
        self.figure.set_size_inches(
            1200 / dpi,
            800 / dpi,
        )

        # Set the axis range
        self.axes.set_xlim(
            0,
            self.size_x,
        )
        self.axes.set_ylim(
            0,
            self.size_y,
        )

        # Move xaxis ticks to the top of the plot
        self.axes.xaxis.tick_top()
        self.axes.invert_yaxis()

        # Set aspect ratio to 1
        self.axes.set_aspect("equal")

        if show_grid:
            self.axes.set_xticks(
                np.arange(
                    start=0.0 if self.delta_x is None else self.delta_x / 2,
                    stop=self.size_x,
                    step=0.5 if self.delta_x is None else self.delta_x,
                )
            )

            self.axes.set_yticks(
                np.arange(
                    start=0.0 if self.delta_y is None else self.delta_y / 2,
                    stop=self.size_x,
                    step=0.5 if self.delta_y is None else self.delta_y,
                )
            )
            self.axes.grid(True)

        if show_sources:
            # Plot the audio sources
            for source in self.sources.values():
                source.plot(ax=self.axes)

        if show_mics:
            # Plot the microphones
            for mic in self.microphones.values():
                mic.plot(ax=self.axes)

        return self.figure, self.axes
