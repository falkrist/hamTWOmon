"""
@author: madengr
"""

from gnuradio import gr  # type: ignore
from gnuradio import filter as grfilter # Don't redefine Python's filter()
from gnuradio.fft import window  # type: ignore
from gnuradio import analog
from gnuradio.filter import pfb  # type: ignore
from gnuradio import blocks
import ctcss_tones as ct
import logging
from typing import Callable

from demodulators.BaseTuner import BaseTuner
from classification import Classifier
from channel_loggers import ChannelLogger

class TunerDemodWBFM(BaseTuner):
    """Tuner, demodulator, and recorder chain for wide band FM demodulation

    Kept as it's own class so multiple can be instantiated in parallel
    Accepts complex baseband samples at 1 Msps minimum
    Frequency translating FIR filter tunes from -samp_rate/2 to +samp_rate/2
    The following sample rates assume 1 Msps input
    First two stages of decimation are 4 each for a total of 16
    Thus first two stages brings 1 Msps down to 40 ksps
    The third stage decimates by int(samp_rate/1E6)
    Thus output rate will vary from 40 ksps to 79.99 ksps
    The channel is filtered to 25.0 KHz bandwidth followed by squelch
    The squelch is non-blocking since samples will be added with other demods
    The quadrature demod is followed by a fourth stage of decimation by 4
    This brings the sample rate down to 8 ksps to 15.98 ksps
    The audio is low-pass filtered to 3.5 kHz bandwidth
    The polyphase resampler resamples by samp_rate/(decims[1] * decims[0]**3)
    Audio rate is configurable at 8 or 16 ksps
    This results in a constant 8/16 ksps, irrespective of RF sample rate
    The audio may then be CTCSS squelch blocked if configured, with a tone
    The audio may then be high-pass filtered to remove CTCSS tones if configured
    This 8/16 ksps audio stream may be added to other demod streams
    The audio is run through an additional blocking squelch at -200 dB
    This stops the sample flow so squelched audio is not recorded to file
    The wav file sink stores 8-bit samples (default/grainy quality but compact)
    Default demodulator center frequency is 0 Hz
    This is desired since hardware DC removal reduces sensitivity at 0 Hz
    WBFM demod of LO leakage will just be 0 amplitude

    Args:
        samp_rate (float): Input baseband sample rate in sps (1E6 minimum)
        audio_rate (float): Output audio sample rate in sps (8 kHz minimum)
        record (bool): Record audio to file if True
        audio_bps (int): Audio bit depth in bps (bits/samples)
        min_file_size (int): Minimum saved wav file size
        ctcss_filter (bool): Filter on set CTCSS tone if True
        ctcss_tone_block (bool): Prevent CTCSS tones in audio output if True

    Attributes:
        center_freq (float): Baseband center frequency in Hz
        record (bool): Record audio to file if True
        time_stamp (int): Time stamp of demodulator start for timing run length
        ctcss_filter (bool): Filter on set CTCSS tone if True
        ctcss_tone (float): CTCSS tone frequency for filter in Hz
        ctcss_index (int): CTCSS tone list index for selected tone
        ctcss_tone_block (bool): Prevent CTCSS tones in audio output if True
        ctcss_level (float): CTCSS tone level required to break squelch (open)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, samp_rate: int, audio_rate: int, record: bool,
                 audio_bps: int, min_recording: float, classify: Classifier | None,
                 notify_scanner: Callable):
        gr.hier_block2.__init__(self, "TunerDemodWBFM",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                                gr.io_signature(1, 1, gr.sizeof_float))

        super().__init__(audio_rate, audio_bps, min_recording, classify, notify_scanner)

        # Default values
        self.center_freq = 0
        squelch_db = -60
        self.quad_demod_gain = 0.050
        self.file_name = None
        self.record = record

        # Decimation values for four stages of decimation
        decims = (5, int(samp_rate/1E6))

        # Low pass filter taps for decimation by 5
        low_pass_filter_taps_0 = \
            grfilter.firdes.low_pass(1, 1, 0.090, 0.010,
                    window.WIN_HAMMING)

        # Frequency translating FIR filter decimating by 5
        self.freq_xlating_fir_filter_ccc = \
            grfilter.freq_xlating_fir_filter_ccc(decims[0],
                                                 low_pass_filter_taps_0,
                                                 self.center_freq, samp_rate)

        # FIR filter decimating by 5
        fir_filter_ccc_0 = grfilter.fir_filter_ccc(decims[0],
                                                   low_pass_filter_taps_0)

        # Low pass filter taps for decimation from samp_rate/25 to 40-79.9 ksps
        # In other words, decimation by int(samp_rate/1E6)
        # 12.5 kHz cutoff for NBFM channel bandwidth
        low_pass_filter_taps_1 = grfilter.firdes.low_pass(
            1, samp_rate/decims[0]**2, 12.5E3, 1E3, window.WIN_HAMMING)

        # FIR filter decimation by int(samp_rate/1E6)
        fir_filter_ccc_1 = grfilter.fir_filter_ccc(decims[1],
                                                   low_pass_filter_taps_1)

        # Non blocking power squelch
        self.analog_pwr_squelch_cc = analog.pwr_squelch_cc(squelch_db,
                                                           1e-1, 0, False)

        # Quadrature demod with gain set for decent audio
        # The gain will be later multiplied by the 0 dB normalized volume
        self.analog_quadrature_demod_cf = \
            analog.quadrature_demod_cf(self.quad_demod_gain)

        # 3.5 kHz cutoff for audio bandwidth
        low_pass_filter_taps_2 = grfilter.firdes.low_pass(1,\
                        samp_rate/(decims[1] * decims[0]**2),\
                        3.5E3, 500, window.WIN_HAMMING)

        # FIR filter decimating by 5 from 40-79.9 ksps to 8-15.98 ksps
        fir_filter_fff_0 = grfilter.fir_filter_fff(decims[0],
                                                   low_pass_filter_taps_2)

        # Polyphase resampler allows arbitary RF sample rates
        # Takes 8-15.98 ksps to a constant 8 ksps for audio
        pfb_resamp = audio_rate/float(samp_rate/(decims[1] * decims[0]**3))
        pfb_arb_resampler_fff = pfb.arb_resampler_fff(pfb_resamp, taps=None,
                                                      flt_size=32)

        # Connect the blocks for the demod
        self.connect(self, self.freq_xlating_fir_filter_ccc)
        self.connect(self.freq_xlating_fir_filter_ccc, fir_filter_ccc_0)
        self.connect(fir_filter_ccc_0, fir_filter_ccc_1)
        self.connect(fir_filter_ccc_1, self.analog_pwr_squelch_cc)
        self.connect(self.analog_pwr_squelch_cc,
                     self.analog_quadrature_demod_cf)
        self.connect(self.analog_quadrature_demod_cf, fir_filter_fff_0)
        self.connect(fir_filter_fff_0, pfb_arb_resampler_fff)
        self.connect(pfb_arb_resampler_fff, self)

        # Need to set this to a very low value of -200 since it is after demod
        # Only want it to gate when the previous squelch has gone to zero
        analog_pwr_squelch_ff = analog.pwr_squelch_ff(-200, 1e-1, 0, True)

        # Connect the blocks for recording
        if (self.record):
            if self.audio_bps == 16:
                wav_format = blocks.FORMAT_PCM_16
            elif self.audio_bps == 8:
                wav_format = blocks.FORMAT_PCM_08
            else:
                wav_format = blocks.FORMAT_PCM_16

            self.blocks_wavfile_sink = blocks.wavfile_sink('/dev/null', 1,
                                                       audio_rate,
                                                       blocks.FORMAT_WAV,
                                                       wav_format,
                                                       False)
            self.connect(pfb_arb_resampler_fff, analog_pwr_squelch_ff)
            self.connect(analog_pwr_squelch_ff, self.blocks_wavfile_sink)
        else:
            null_sink1 = blocks.null_sink(gr.sizeof_float)
            self.connect(pfb_arb_resampler_fff, analog_pwr_squelch_ff)
            self.connect(analog_pwr_squelch_ff, null_sink1)

    def set_volume(self, volume_db):
        """Sets the volume

        Args:
            volume_db (float): Volume in dB
        """
        gain = self.quad_demod_gain * 10**(volume_db/20.0)
        self.analog_quadrature_demod_cf.set_gain(gain)

    def set_ctcss_tone(self, ctcss_tone):
        """Sets the CTCSS tone frequency by selecting the index from tone list that matches the input value

        Args:
            ctcss_tone (float): CTCSS tone frequency in Hz
        """
        self.ctcss_index = ct.ctcss_tones.index(ctcss_tone)
        if (~self.ctcss_index):
            self.ctcss_index = 0
        self.ctcss_tone = ct.ctcss_tones[self.ctcss_index]
        return self.ctcss_tone
