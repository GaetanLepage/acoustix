"""
Room using the GPU RIR backend.
https://github.com/DavidDiazGuerra/gpuRIR
https://arxiv.org/abs/1810.11359
"""

from dataclasses import dataclass
from typing import Optional

import exputils as eu
import numpy as np

from .room import Room


@dataclass
class _SimulationParams:
    beta: np.ndarray

    # Time to start the diffuse reverberation model [s]
    time_diffuse_reverb: float

    # Number of image sources in each dimension
    nb_img: list[int]


class GpuRirRoom(Room):
    """
    Implementation of the abstract Room class using the GpuRIR library as backend.
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

        self._simulation_params: Optional[_SimulationParams] = None

    def _init_simulation_params(self) -> None:
        """
        Should be called only once.
        Initialize the RIR simulation parameters.
        This is out of the __init__ method to avoid importing the gpuRIR library while not needing
        to simulate anything.
        """
        # Lazily importing GpuRIR to avoid CUDA GPU initialization errors when using
        # multiprocessing.
        import gpuRIR

        gpuRIR.activateMixedPrecision(False)
        gpuRIR.activateLUT(True)

        self._logger.info("Initializing simulation params for gpuRirRoom")

        # Size of the room [m]
        room_dim: np.ndarray = self.get_room_dims()

        # Absortion coefficient ratios of the walls
        abs_weights: list[float] = [0.9] * 5 + [0.5]

        # Attenuation when start using the diffuse reverberation model [dB]
        att_diff: float = 15.0

        # Attenuation at the end of the simulation [dB]
        att_max: float = 60.0

        # Reflection coefficients
        beta: np.ndarray = gpuRIR.beta_SabineEstimation(
            room_sz=room_dim,
            T60=self.rt_60,
            abs_weights=abs_weights,
        )

        # Time to start the diffuse reverberation model [s]
        time_diffuse_reverb: float = gpuRIR.att2t_SabineEstimator(
            att_dB=att_diff,
            T60=self.rt_60,
        )
        if self._max_time_auto:
            self.max_time = gpuRIR.att2t_SabineEstimator(
                att_dB=att_max,
                T60=self.rt_60,
            )
        # Number of image sources in each dimension
        nb_img: list[int] = gpuRIR.t2n(
            T=time_diffuse_reverb,
            rooms_sz=room_dim,
        )

        # Initialize the simulation parameters:
        self._simulation_params = _SimulationParams(
            beta=beta,
            time_diffuse_reverb=time_diffuse_reverb,
            nb_img=nb_img,
        )

    def pre_compute_rir(self) -> None:
        """
        Pre compute the RIR for every source-microphone pair.
        """
        import gpuRIR

        if self.rir_up_to_date:
            # self._logger.debug("RIR is already up to date: returning.")
            return
        # self._logger.debug("Computing RIR")

        if not self._simulation_params:
            self._init_simulation_params()
        assert self._simulation_params is not None

        # Size of the room [m]
        room_dim: np.ndarray = np.array(
            [
                self.size_x,
                self.size_y,
                self.height,
            ],
        )

        assert len(self.sources) > 0

        # Sources
        sources_positions: np.ndarray = np.array(
            [source.position for source in self.sources.values()]
        )
        sources_orientations: np.ndarray = np.array(
            [source.orientation for source in self.sources.values()]
        )
        source_pattern: str = self.get_source_from_index(
            index=0,
        ).pattern

        # Microphones
        mic_positions: np.ndarray = np.array(
            [mic.position for mic in self.microphones.values()],
        )
        mic_orientations: np.ndarray = np.array(
            [mic.orientation for mic in self.microphones.values()]
        )
        mic_pattern: str = self.get_mic_from_index(
            index=0,
        ).pattern

        self.rir: np.ndarray = gpuRIR.simulateRIR(
            room_sz=room_dim,
            beta=self._simulation_params.beta,
            # Sources
            pos_src=sources_positions,
            spkr_pattern=source_pattern,
            orV_src=sources_orientations,
            # Microphones
            pos_rcv=mic_positions,
            mic_pattern=mic_pattern,
            orV_rcv=mic_orientations,
            nb_img=self._simulation_params.nb_img,
            Tmax=self.max_time,
            fs=self.sampling_frequency,
            Tdiff=self._simulation_params.time_diffuse_reverb,
        )

        # We have to transpose the result to have microphones on the first axis and sources on the
        # second axis.
        self.rir = self.rir.transpose(1, 0, 2)

        self.rir_up_to_date = True
