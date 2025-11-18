"""
Room using the PyRoomAcoustics backend.
"""

import exputils as eu
import numpy as np
import pyroomacoustics as pra

from .room import Microphone, Room, Source


class PyRoomAcousticsRoom(Room):
    """
    Implementation of the abstract Room class using the pyroomacoustics library as backend.
    """

    def __init__(
        self,
        config: eu.AttrDict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )

        # Initializing the PyRoomAcoustics shoebox room
        room_dim: tuple[float, float, float] = (
            self.size_x,
            self.size_y,
            self.height,
        )

        e_absorption: float
        max_order: int
        material: pra.Material
        if self.rt_60 > 0:
            e_absorption, max_order = pra.inverse_sabine(
                rt60=self.rt_60,
                room_dim=room_dim,
            )

            material = pra.Material(
                energy_absorption=e_absorption,
            )
        else:
            material = pra.Material(
                energy_absorption="hard_surface",
            )
            max_order = 17

        self.pa_room: pra.ShoeBox = pra.ShoeBox(
            p=room_dim,
            fs=self.sampling_frequency,
            materials=material,
            max_order=max_order,
        )

    def pre_compute_rir(self) -> None:
        """
        Precompute the RIR for all sampled grid positions and orientations.
        """
        self.pa_room.compute_rir()

        # Not necessary if we use PyRoomAcoustics' internal way of computing audio signal.
        # Indeed, the RIR matrix is already stored in the pa_room attribute.
        self.rir = self.pa_room.rir

        self.rir_up_to_date = True

    def add_source(
        self,
        source: Source,
        clip_to_grid: bool = False,
    ) -> None:
        super().add_source(
            source=source,
        )
        self.pa_room.add_source(
            position=source.position,
        )

    def add_microphone(
        self,
        mic: Microphone,
        clip_to_grid: bool = False,
    ) -> None:
        super().add_microphone(
            mic=mic,
        )
        self.pa_room.add_microphone(
            loc=mic.position,
            fs=mic.sampling_frequency,
        )

    def add_microphones(
        self,
        microphones_list: list[Microphone],
        clip_to_grid: bool = False,
    ) -> None:
        # assert self.pa_room.n_mics == 0

        # mic_locations_3d: np.ndarray = np.array([[mic_x, mic_y, height]
        #                                          for mic_x, mic_y
        #                                          in mic_locations])

        # mic_locations_3d = mic_locations_3d.T

        # self.pa_room.add_microphone_array(mic_array=mic_locations_3d)
        raise NotImplementedError

    def set_source_input_audio_signal(
        self,
        input_audio_signal: np.ndarray,
        delay: float = 0,
        source_name: str = "",
        source_index: int = -1,
    ) -> None:
        super().set_source_input_audio_signal(
            input_audio_signal=input_audio_signal,
            delay=delay,
            source_name=source_name,
            source_index=source_index,
        )

        if source_index < 0:
            source_index = self.get_source_index(
                source_name=source_name,
            )

        self.pa_room.sources[source_index].add_signal(
            signal=input_audio_signal,
        )

    def clear_sources(self) -> None:
        """
        Removes all the audio sources from the room.
        """
        super().clear_sources()
        self.pa_room.sources = []


#     def simulate(self) -> None:
#         if self.simulation_up_to_date:
#             print("Useless call to `simulate()`: the simulation is already up to date.")
#             return
#
#         if not self.rir_up_to_date:
#             self.pre_compute_rir()
#
#         self.pa_room.simulate()
#         for mic_index, mic in enumerate(self.microphones.values()):
#             raw_signal = self.pa_room.mic_array.signals[mic_index]
#             mic.listened_signal = np.array(pra.utilities.normalize(signal=raw_signal,
#                                                                    bits=16),
#                                            dtype=np.int16)
#
#         self.simulation_up_to_date = True

#     def get_audio(self,
#                   source_audio_signals: List[np.ndarray]) -> List[np.ndarray]:
#         assert len(source_audio_signals) == len(self.sources)
#
#         if not self.rir_up_to_date:
#             self.pre_compute_rir()
#
#
#         raw_signal: np.ndarray
#         normalized_signal: np.ndarray
#         mic_audio_signals: List[np.ndarray] = []

#     def get_audio_at_mic(self,
#                          mic_name: str = '',
#                          mic_index: int = -1) -> np.ndarray:
#         if not self.rir_up_to_date:
#             self.pre_compute_rir()
#
#         self.pa_room.simulate()
#
#         if mic_index == - 1:
#             assert mic_name != '', "Either a mic index or a mic name should be provided."
#             mic_index = self.get_mic_index(mic_name=mic_name)
#
#         raw_signal: np.ndarray = self.pa_room.mic_array.signals[mic_index]
#
#         normalized_signal: np.ndarray = pra.utilities.normalize(signal=raw_signal,
#                                                                 bits=16)
#         return np.array(normalized_signal, dtype=np.int16)
