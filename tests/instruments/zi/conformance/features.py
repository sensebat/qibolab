"""What qibolab needs from a controller so that qibocal can calibrate.

One row per feature. ``checkpoint`` is the first of the checkpoints CP1-CP5 (D27, D28
in the ZI driver plan) that needs the feature, or ``"beyond"`` when no checkpoint needs
it yet. ``qibocal`` names protocols that use the feature (paths are relative to
``qibocal/src/qibocal/protocols``). ``laboneq`` names the LabOne Q construct that can
implement it. ``effects`` lists the effect tests of the feature, as
``<file>::<function>`` in this folder: what a qibolab call makes LabOne Q do (or the
refusal it gets), observed at the LabOne Q boundary and run under planted driver bugs
(``test_effects_*.py``). An empty list means no effect test yet; the contract tests of
``test_conformance.py`` still cover the feature. test_effects_smoke.py checks that this
field and the ``feature`` markers agree.
"""

FEATURES = [
    # sweep
    {
        "id": "F01",
        "title": "Frequency sweep on a probe channel",
        "group": "sweep",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:205), resonator_punchout, dispersive_shift",
        "laboneq": "SweepParameter on the probe oscillator frequency",
        "effects": [
            "test_effects_configs_errors.py::test_frequency_sweep_on_acquisition_channel",
            "test_effects_sweeps.py::test_probe_frequency_sweep_keeps_other_probe",
            "test_effects_sweeps.py::test_probe_frequency_sweep_values",
        ],
    },
    {
        "id": "F02",
        "title": "Frequency sweep on a drive channel",
        "group": "sweep",
        "checkpoint": "CP3",
        "qibocal": "qubit_spectroscopy (qubit_spectroscopies/qubit_spectroscopy.py:196), rabi_amplitude_frequency_signal",
        "laboneq": "SweepParameter on the drive oscillator frequency",
        "effects": [
            "test_effects_sweeps.py::test_drive_frequency_sweep_results",
            "test_effects_sweeps.py::test_drive_frequency_sweep_values",
        ],
    },
    {
        "id": "F03",
        "title": "Amplitude sweep on a readout probe",
        "group": "sweep",
        "checkpoint": "CP2",
        "qibocal": "resonator_punchout (resonator_spectroscopies/resonator_punchout.py:123)",
        "laboneq": "play(probe, amplitude=SweepParameter)",
        "effects": [
            "test_effects_sweeps.py::test_probe_amplitude_sweep",
        ],
    },
    {
        "id": "F04",
        "title": "Amplitude sweep on a drive pulse",
        "group": "sweep",
        "checkpoint": "CP4",
        "qibocal": "rabi_amplitude_signal (rabi/amplitude_signal.py:91), rabi_amplitude (rabi/amplitude.py:66)",
        "laboneq": "play(drive, amplitude=SweepParameter)",
        "effects": [
            "test_effects_sweeps.py::test_drive_amplitude_sweep",
        ],
    },
    {
        "id": "F05",
        "title": "Amplitude sweep on a flux pulse",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "chevron (two_qubit_interaction/chevron/chevron.py:189), flux_amplitude_frequency (flux_dependence/flux_amplitude_frequency.py:231)",
        "laboneq": "play(flux, amplitude=SweepParameter) on the HDAWG",
        "effects": [],
    },
    {
        "id": "F06",
        "title": "Duration sweep on a drive pulse and the delay before readout",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "rabi_length_signal (rabi/length_signal.py:97)",
        "laboneq": "play(drive, length=SweepParameter) and delay(time=SweepParameter)",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_sweep_parameter",
        ],
    },
    {
        "id": "F07",
        "title": "Duration sweep on a delay",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "t1 (coherence/t1.py:63), t1_signal, ramsey (ramsey/acquisition.py:213), spin_echo",
        "laboneq": "delay(signal, time=SweepParameter)",
        "effects": [],
    },
    {
        "id": "F08",
        "title": "Duration sweep on a flux pulse",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "chevron (two_qubit_interaction/chevron/chevron.py:194), cryoscope (flux_dependence/cryoscope.py:370)",
        "laboneq": "play(flux, length=SweepParameter) on the HDAWG",
        "effects": [],
    },
    {
        "id": "F09",
        "title": "Interpolated duration sweep on a drive pulse",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "rabi_length_signal (rabi/length_signal.py:91), cross_resonance_length",
        "laboneq": "pulse functional with a swept length, samples stretched",
        "effects": [],
    },
    {
        "id": "F10",
        "title": "Relative phase sweep on a drive pulse",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "qubit_vz (flux_dependence/qubit_vz.py:121), cancellation_phase",
        "laboneq": "play(drive, phase=SweepParameter)",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_sweep_parameter",
        ],
    },
    {
        "id": "F11",
        "title": "Phase sweep on a virtual Z",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "standard_rb_sweeper (randomized_benchmarking/standard_rb_sweeper.py:132), virtual_z_phases (two_qubit_interaction/virtual_z_phases.py:251)",
        "laboneq": "play(increment_oscillator_phase=SweepParameter)",
        "effects": [],
    },
    {
        "id": "F12",
        "title": "Offset sweep on a flux channel",
        "group": "sweep",
        "checkpoint": "beyond",
        "qibocal": "qubit_flux_dependence (flux_dependence/qubit_flux_dependence.py:149), resonator_flux_dependence (flux_dependence/resonator_flux_dependence.py:129)",
        "laboneq": "SignalCalibration.voltage_offset=SweepParameter (near time)",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_sweep_parameter",
        ],
    },
    {
        "id": "F13",
        "title": "Several sweepers in parallel, one per qubit",
        "group": "sweep",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:204)",
        "laboneq": "one sweep() over several SweepParameters",
        "effects": [
            "test_effects_sweeps.py::test_parallel_drive_frequency_sweepers",
            "test_effects_sweeps.py::test_parallel_probe_frequency_sweepers",
        ],
    },
    {
        "id": "F14",
        "title": "Two nested sweep dimensions",
        "group": "sweep",
        "checkpoint": "CP2",
        "qibocal": "resonator_punchout (resonator_spectroscopies/resonator_punchout.py:137)",
        "laboneq": "nested sweep() blocks",
        "effects": [
            "test_effects_acquisition.py::test_result_axes_follow_sweepers",
            "test_effects_sweeps.py::test_nested_drive_sweepers_grid",
            "test_effects_sweeps.py::test_nested_probe_sweepers_grid",
        ],
    },
    {
        "id": "F15",
        "title": "One sweeper over the pulses of several qubits",
        "group": "sweep",
        "checkpoint": "CP2",
        "qibocal": "resonator_punchout (resonator_spectroscopies/resonator_punchout.py:126), rabi_amplitude_signal (rabi/amplitude_signal.py:94)",
        "laboneq": "one SweepParameter used by several play() calls",
        "effects": [
            "test_effects_sweeps.py::test_one_amplitude_sweeper_two_drives",
            "test_effects_sweeps.py::test_one_amplitude_sweeper_two_probes",
        ],
    },
    # acquisition
    {
        "id": "F16",
        "title": "Integrated acquisition (INTEGRATION)",
        "group": "acquisition",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:226) and every *_signal protocol",
        "laboneq": "AcquisitionType.INTEGRATION",
        "effects": [
            "test_effects_acquisition.py::test_integration_reaches_labone_q",
        ],
    },
    {
        "id": "F17",
        "title": "State discrimination (DISCRIMINATION)",
        "group": "acquisition",
        "checkpoint": "beyond",
        "qibocal": "rabi_amplitude, t1 (coherence/t1.py:75), flipping, drag",
        "laboneq": "AcquisitionType.DISCRIMINATION with the SHFQA threshold",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_acquisition_type",
        ],
    },
    {
        "id": "F18",
        "title": "Raw traces (RAW)",
        "group": "acquisition",
        "checkpoint": "beyond",
        "qibocal": "time_of_flight_readout (signal_experiments/time_of_flight_readout.py:102), calibrate_state_discrimination (signal_experiments/calibrate_state_discrimination.py:90)",
        "laboneq": "AcquisitionType.RAW",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_acquisition_type",
        ],
    },
    # averaging
    {
        "id": "F19",
        "title": "Cyclic averaging (CYCLIC)",
        "group": "averaging",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:227), punchout, qubit_spectroscopy, rabi_amplitude_signal",
        "laboneq": "AveragingMode.CYCLIC",
        "effects": [
            "test_effects_acquisition.py::test_averaging_mode_reaches_labone_q",
        ],
    },
    {
        "id": "F20",
        "title": "Single shots (SINGLESHOT)",
        "group": "averaging",
        "checkpoint": "CP5",
        "qibocal": "single_shot_classification (classification/classification.py:245)",
        "laboneq": "AveragingMode.SINGLE_SHOT",
        "effects": [
            "test_effects_acquisition.py::test_averaging_mode_reaches_labone_q",
        ],
    },
    {
        "id": "F21",
        "title": "Sequential averaging (SEQUENTIAL)",
        "group": "averaging",
        "checkpoint": "beyond",
        "qibocal": "none",
        "laboneq": "AveragingMode.SEQUENTIAL",
        "effects": [
            "test_effects_configs_errors.py::test_unsupported_averaging_mode",
        ],
    },
    {
        "id": "F46",
        "title": "Single shots with a sweeper",
        "group": "averaging",
        "checkpoint": "beyond",
        "qibocal": "t1_signal (coherence/t1_signal.py:141), ro_amplitude_frequency (readout/amplitude_frequency_optimization.py:239)",
        "laboneq": "AveragingMode.SINGLE_SHOT with sweeps, shots axis first",
        "effects": [
            "test_effects_acquisition.py::test_result_axes_follow_sweepers",
        ],
    },
    # sequence
    {
        "id": "F22",
        "title": "Readout: probe and acquisition as one event",
        "group": "sequence",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:192) and every protocol through the MZ native",
        "laboneq": "play(probe) + acquire(acquisition)",
        "effects": [
            "test_effects_acquisition.py::test_integration_length_is_acquisition_duration",
            "test_effects_pulses.py::test_readout_is_probe_and_acquisition",
        ],
    },
    {
        "id": "F23",
        "title": "Readout of several qubits at once",
        "group": "sequence",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:191), all multiplexed protocols",
        "laboneq": "one acquire() per qubit on the same SHFQA channel",
        "effects": [
            "test_effects_acquisition.py::test_multiplexed_results_reach_their_readout",
            "test_effects_pulses.py::test_multiplexed_readouts_on_their_channels",
        ],
    },
    {
        "id": "F24",
        "title": "Drive pulse, then readout after a delay",
        "group": "sequence",
        "checkpoint": "CP3",
        "qibocal": "qubit_spectroscopy (qubit_spectroscopies/qubit_spectroscopy.py:188), rabi (rabi/utils.py:326)",
        "laboneq": "sections in order, delay(time=...)",
        "effects": [
            "test_effects_pulses.py::test_amplitude_and_phase_reach_their_drive",
            "test_effects_pulses.py::test_delay_shifts_pulse_and_readout",
            "test_effects_smoke.py::test_drive_amplitude_reaches_drive_port",
        ],
    },
    {
        "id": "F25",
        "title": "Probe pulse and Acquisition as separate events",
        "group": "sequence",
        "checkpoint": "beyond",
        "qibocal": "none directly (platforms may write the MZ native this way)",
        "laboneq": "separate play(probe) and acquire()",
        "effects": [],
    },
    {
        "id": "F26",
        "title": "Align between channels",
        "group": "sequence",
        "checkpoint": "beyond",
        "qibocal": "rabi_length_signal with interpolated sweeper (rabi/utils.py:364)",
        "laboneq": "section boundaries across signals",
        "effects": [
            "test_effects_pulses.py::test_align_synchronises_channels",
        ],
    },
    {
        "id": "F27",
        "title": "Virtual Z between drive pulses",
        "group": "sequence",
        "checkpoint": "beyond",
        "qibocal": "standard_rb_sweeper (randomized_benchmarking/standard_rb_sweeper.py:82), natives with phases",
        "laboneq": "play(increment_oscillator_phase=...)",
        "effects": [],
    },
    {
        "id": "F28",
        "title": "Pulse on a flux channel",
        "group": "sequence",
        "checkpoint": "beyond",
        "qibocal": "flux_gate (flux_dependence/flux_gate.py:121), chevron, cryoscope",
        "laboneq": "play() on an HDAWG flux line",
        "effects": [],
    },
    {
        "id": "F29",
        "title": "Several sequences in one call",
        "group": "sequence",
        "checkpoint": "CP5",
        "qibocal": "single_shot_classification (classification/classification.py:245)",
        "laboneq": "several sections per shot, one handle per acquisition",
        "effects": [
            "test_effects_acquisition.py::test_relaxation_time_after_each_sequence",
            "test_effects_acquisition.py::test_results_of_several_sequences",
            "test_effects_pulses.py::test_several_sequences_in_order",
        ],
    },
    {
        "id": "F30",
        "title": "Several sequences in one call, with a sweeper",
        "group": "sequence",
        "checkpoint": "CP5",
        "qibocal": "dispersive_shift (dispersive_shift/dispersive_shift.py:131)",
        "laboneq": "several sections per shot inside one sweep()",
        "effects": [],
    },
    {
        "id": "F47",
        "title": "Two readouts of one qubit in one sequence",
        "group": "sequence",
        "checkpoint": "beyond",
        "qibocal": "readout_characterization (readout/readout_characterization.py:101), ro_amplitude_frequency (readout/amplitude_frequency_optimization.py:173)",
        "laboneq": "two acquire() on one signal, one handle each",
        "effects": [],
    },
    # envelope
    {
        "id": "F31",
        "title": "Rectangular envelope",
        "group": "envelope",
        "checkpoint": "CP1",
        "qibocal": "resonator_spectroscopy (resonator_spectroscopies/resonator_spectroscopy.py:197), qubit_spectroscopy",
        "laboneq": "pulse_library.const",
        "effects": [
            "test_effects_pulses.py::test_duration_played",
            "test_effects_pulses.py::test_envelope_samples_played",
        ],
    },
    {
        "id": "F32",
        "title": "Gaussian envelope",
        "group": "envelope",
        "checkpoint": "CP4",
        "qibocal": "rabi_amplitude_signal through the RX native (rabi/utils.py:313)",
        "laboneq": "pulse_library.gaussian or a pulse functional",
        "effects": [
            "test_effects_pulses.py::test_envelope_samples_played",
        ],
    },
    {
        "id": "F33",
        "title": "DRAG envelope",
        "group": "envelope",
        "checkpoint": "beyond",
        "qibocal": "drag_tuning (drag/drag.py:135), allxy (allxy/allxy.py:135)",
        "laboneq": "pulse_library.drag or a pulse functional",
        "effects": [
            "test_effects_pulses.py::test_envelope_samples_played",
            "test_effects_pulses.py::test_iq_not_swapped",
        ],
    },
    {
        "id": "F34",
        "title": "Gaussian square envelope",
        "group": "envelope",
        "checkpoint": "beyond",
        "qibocal": "none directly (platform natives, such as cross resonance and flux pulses)",
        "laboneq": "pulse_library.gaussian_square or a pulse functional",
        "effects": [
            "test_effects_pulses.py::test_envelope_samples_played",
        ],
    },
    {
        "id": "F35",
        "title": "Exponential envelope",
        "group": "envelope",
        "checkpoint": "beyond",
        "qibocal": "none directly (platform natives for flux pulses)",
        "laboneq": "pulse functional",
        "effects": [
            "test_effects_pulses.py::test_envelope_samples_played",
        ],
    },
    {
        "id": "F36",
        "title": "SNZ envelope",
        "group": "envelope",
        "checkpoint": "beyond",
        "qibocal": "snz_optimize (two_qubit_interaction/snz_optimize.py:169)",
        "laboneq": "pulse functional",
        "effects": [
            "test_effects_pulses.py::test_envelope_samples_played",
        ],
    },
    {
        "id": "F37",
        "title": "Custom envelope (sampled)",
        "group": "envelope",
        "checkpoint": "beyond",
        "qibocal": "none directly (platform natives)",
        "laboneq": "sampled_pulse_complex",
        "effects": [
            "test_effects_pulses.py::test_custom_envelope_samples_played",
        ],
    },
    # config
    {
        "id": "F38",
        "title": "DC offset on a flux channel",
        "group": "config",
        "checkpoint": "CP1",
        "qibocal": "qubit_flux_dependence reads it (flux_dependence/qubit_flux_dependence.py:146)",
        "laboneq": "SignalCalibration.voltage_offset",
        "effects": [
            "test_effects_configs_errors.py::test_dc_offset_reaches_hdawg_port",
            "test_effects_configs_errors.py::test_missing_channel_config",
            "test_effects_configs_errors.py::test_wrong_config_kind",
        ],
    },
    {
        "id": "F39",
        "title": "Filters on a flux channel (exponential, FIR)",
        "group": "config",
        "checkpoint": "beyond",
        "qibocal": "cryoscope clears them (flux_dependence/cryoscope.py:385)",
        "laboneq": "SignalCalibration.precompensation",
        "effects": [],
    },
    {
        "id": "F40",
        "title": "Channel frequency and LO frequency, LO moved by an update",
        "group": "config",
        "checkpoint": "CP3",
        "qibocal": "qubit_spectroscopy LO batching (qubit_spectroscopies/qubit_spectroscopy.py:212)",
        "laboneq": "SignalCalibration oscillator and local_oscillator",
        "effects": [
            "test_effects_configs_errors.py::test_iq_frequency_is_lo_plus_oscillator",
            "test_effects_configs_errors.py::test_lo_update_keeps_channel_frequency",
            "test_effects_configs_errors.py::test_missing_lo_config",
            "test_effects_configs_errors.py::test_wrong_config_kind",
        ],
    },
    {
        "id": "F41",
        "title": "Integration kernel in the acquisition config",
        "group": "config",
        "checkpoint": "beyond",
        "qibocal": "calibrate_state_discrimination writes it (signal_experiments/calibrate_state_discrimination.py:237)",
        "laboneq": "acquire(kernel=sampled pulse)",
        "effects": [
            "test_effects_configs_errors.py::test_kernel_refused",
        ],
    },
    {
        "id": "F42",
        "title": "Acquisition delay (time of flight) changed by an update",
        "group": "config",
        "checkpoint": "CP1",
        "qibocal": "time_of_flight_readout (signal_experiments/time_of_flight_readout.py:106)",
        "laboneq": "SignalCalibration.port_delay",
        "effects": [
            "test_effects_acquisition.py::test_acquisition_delay_follows_config",
            "test_effects_configs_errors.py::test_acquisition_delay_delays_integration",
            "test_effects_configs_errors.py::test_wrong_config_kind",
        ],
    },
    {
        "id": "F43",
        "title": "New configs on each call, same connection",
        "group": "config",
        "checkpoint": "CP3",
        "qibocal": "qubit_spectroscopy batches (qubit_spectroscopies/qubit_spectroscopy.py:218), rabi_amplitude_signal updates (rabi/amplitude_signal.py:103)",
        "laboneq": "new Experiment calibration per run on one Session",
        "effects": [
            "test_effects_configs_errors.py::test_each_call_uses_its_own_configs",
            "test_effects_configs_errors.py::test_missing_channel_config",
        ],
    },
    # execution
    {
        "id": "F44",
        "title": "Number of shots and relaxation time",
        "group": "execution",
        "checkpoint": "CP1",
        "qibocal": "every protocol passes nshots and relaxation_time (resonator_spectroscopies/resonator_spectroscopy.py:224)",
        "laboneq": "acquire_loop_rt(count=nshots, repetition_time=...)",
        "effects": [
            "test_effects_acquisition.py::test_nshots_is_acquire_loop_count",
            "test_effects_acquisition.py::test_relaxation_time_after_each_readout",
        ],
    },
    {
        "id": "F45",
        "title": "Fast reset",
        "group": "execution",
        "checkpoint": "beyond",
        "qibocal": "none",
        "laboneq": "active reset, conditional play on the measured state",
        "effects": [
            "test_effects_configs_errors.py::test_fast_reset_refused",
        ],
    },
]

BY_ID = {f["id"]: f for f in FEATURES}
