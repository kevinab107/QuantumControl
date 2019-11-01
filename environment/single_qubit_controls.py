# Copyright [2019] [Kevin Abraham]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ..exceptions.exceptions import ArgumentsValueError

class SingleQubitControls(object):  
    """
    Creates a control pulse for single qubit operations. The controls are rabi rates
    (for x and y), detuing. Based on the theory of Rotating wave approximation for 
    single qubit control.
    https://qudev.phys.ethz.ch/static/content/QSIT12/QSIT12_Super_L04.pdf
    
    Parameters
    ----------
    rabi_rates : numpy.ndarray, optional
        1-D array of size nx1 where n is number of segments;
        Each entry is the rabi rate for the segment. Defaults to None
    azimuthal_angles : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the azimuthal angle for the segment; Defaults to None
    detunings : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the detuning angle for the segment; Defaults to None
    durations : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the duration of the segment (in seconds); Defaults to None
    name : string, optional
        An optional string to name the  control. Defaults to None.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    def __init__(self,
                 rabi_rates=None,
                 azimuthal_angles=None,
                 detunings=None,
                 durations=None,
                 name=None):

        self.name = name
        if self.name is not None:
            self.name = str(self.name)

        check_none_values = [(rabi_rates is None), (azimuthal_angles is None),
                             (detunings is None), (durations is None)]
        all_are_none = all(value is True for value in check_none_values)
        if all_are_none:
            rabi_rates = np.array([np.pi])
            azimuthal_angles = np.array([0.])
            detunings = np.array([0.])
            durations = np.array([1.])
        else:
            # some may be None while others are not
            input_array_lengths = []
            if not check_none_values[0]:
                rabi_rates = np.array(rabi_rates, dtype=np.float).reshape((-1,))
                input_array_lengths.append(rabi_rates.shape[0])

            if not check_none_values[1]:
                azimuthal_angles = np.array(azimuthal_angles, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(azimuthal_angles))

            if not check_none_values[2]:
                detunings = np.array(detunings, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(detunings))

            if not check_none_values[3]:
                durations = np.array(durations, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(durations))

            # check all valid array lengths are equal
            if max(input_array_lengths) != min(input_array_lengths):
                raise ArgumentsValueError('Rabi rates, Azimuthal angles, Detunings and Durations '
                                          'must be of same length',
                                          {'rabi_rates': rabi_rates,
                                           'azimuthal_angles': azimuthal_angles,
                                           'detunings': detunings,
                                           'durations': durations})

            valid_input_length = max(input_array_lengths)
            if check_none_values[0]:
                rabi_rates = np.zeros((valid_input_length,))
            if check_none_values[1]:
                azimuthal_angles = np.zeros((valid_input_length,))
            if check_none_values[2]:
                detunings = np.zeros((valid_input_length,))
            if check_none_values[3]:
                durations = np.ones((valid_input_length,))

        self.rabi_rates = rabi_rates
        self.azimuthal_angles = azimuthal_angles
        self.detunings = detunings
        self.durations = durations

        
        UPPER_BOUND_RABI_RATE = 1e10
        """Maximum allowed rabi rate
        """

        UPPER_BOUND_DETUNING_RATE = UPPER_BOUND_RABI_RATE
        """Maximum allowed detuning rate
        """

        UPPER_BOUND_DURATION = 1e6
        """Maximum allowed duration of a control
        """

        LOWER_BOUND_DURATION = 1e-12
        """Minimum allowed duration of a control
        """

        UPPER_BOUND_SEGMENTS = 10000
        """Maximum number of segments allowed in a control
        """


        # check if all the rabi_rates are greater than zero
        if np.any(rabi_rates < 0.):
            raise ArgumentsValueError('All rabi rates must be greater than zero.',
                                      {'rabi_rates': rabi_rates},
                                      extras={
                                          'azimuthal_angles': azimuthal_angles,
                                          'detunings': detunings,
                                          'durations': durations})

        # check if all the durations are greater than zero
        if np.any(durations <= 0):
            raise ArgumentsValueError('Duration of driven control segments must all be greater'
                                      + ' than zero.',
                                      {'durations': self.durations})

        if self.number_of_segments > UPPER_BOUND_SEGMENTS:
            raise ArgumentsValueError(
                'The number of segments must be smaller than the upper bound:'
                + str(UPPER_BOUND_SEGMENTS),
                {'number_of_segments': self.number_of_segments})

        if self.maximum_rabi_rate > UPPER_BOUND_RABI_RATE:
            raise ArgumentsValueError(
                'Maximum rabi rate of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_RABI_RATE),
                {'maximum_rabi_rate': self.maximum_rabi_rate})

        if self.maximum_detuning > UPPER_BOUND_DETUNING_RATE:
            raise ArgumentsValueError(
                'Maximum detuning of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DETUNING_RATE),
                {'maximum_detuning': self.maximum_detuning})
        if self.maximum_duration > UPPER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Maximum duration of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DURATION),
                {'maximum_duration': self.maximum_duration})
        if self.minimum_duration < LOWER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Minimum duration of segments must be larger than the lower bound: '
                + str(LOWER_BOUND_DURATION),
                {'minimum_duration': self.minimum_duration})

    @property
    def number_of_segments(self):

        """Returns the number of segments

        Returns
        -------
        int
            The number of segments in the driven control
        """

        return self.rabi_rates.shape[0]

    @property
    def maximum_rabi_rate(self):
        """Returns the maximum rabi rate of the control

        Returns
        -------
        float
            The maximum rabi rate of the control
        """

        return np.amax(self.rabi_rates)

    @property
    def maximum_detuning(self):
        """Returns the maximum detuning of the control

        Returns
        -------
        float
            The maximum detuning of the control
        """
        return np.amax(self.detunings)

    @property
    def amplitude_x(self):
        """Return the X-Amplitude

        Returns
        -------
        numpy.ndarray
            X-Amplitude of each segment
        """

        return self.rabi_rates * np.cos(self.azimuthal_angles)

    @property
    def amplitude_y(self):
        """Return the Y-Amplitude

        Returns
        -------
        numpy.ndarray
            Y-Amplitude of each segment
        """

        return self.rabi_rates * np.sin(self.azimuthal_angles)

    @property
    def directions(self):

        """Returns the directions

        Returns
        -------
        numpy.ndarray
            Directions as 1-D array of floats
        """
        amplitudes = np.sqrt(self.amplitude_x ** 2 +
                             self.amplitude_y ** 2 +
                             self.detunings ** 2)
        normalized_amplitude_x = self.amplitude_x/amplitudes
        normalized_amplitude_y = self.amplitude_y/amplitudes
        normalized_detunings = self.detunings/amplitudes

        normalized_amplitudes = np.hstack((normalized_amplitude_x[:, np.newaxis],
                                           normalized_amplitude_y[:, np.newaxis],
                                           normalized_detunings[:, np.newaxis]))

        directions = np.array([normalized_amplitudes if amplitudes[i] != 0. else
                               np.zeros([3, ]) for i in range(self.number_of_segments)])

        return directions

    @property
    def times(self):
        """Returns the time of each segment within the duration
        of the control

        Returns
        ------
        numpy.ndarray
            Segment times as 1-D array of floats
        """

        return np.insert(np.cumsum(self.durations), 0, 0.)

    @property
    def maximum_duration(self):
        """Returns the maximum duration of all the control segments

        Returns
        -------
        float
            The maximum duration of all the control segments
        """

        return np.amax(self.durations)

    @property
    def minimum_duration(self):
        """Returns the minimum duration of all the control segments

        Returns
        -------
        float
            The minimum duration of all the controls segments
        """

        return np.amin(self.durations)

    @property
    def duration(self):
        """Returns the total duration of the control

        Returns
        -------
        float
            Total duration of the control
        """

        return np.sum(self.durations)



if __name__ == '__main__':
    pass
