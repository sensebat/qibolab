# ZI driver: plan for the missing features

Base: `ElStabilini/qibolab@zi_drivers` (e8757e58), branch `orkun/proto-features`.
Missing items come from the driver audit (ids F01-F47, X01-X26, G1-G7).
Everything below was checked by LabOne Q emulation (`do_emulation=True`) on laboneq 26.4.0 and 26.10.1, same result on both unless noted. Emulation compiles and returns placeholder data; it says nothing about values on hardware.

Paths: `zi/` = `src/qibolab/_core/instruments/zi/`.

## Branches

| branch | worktree | scope |
|---|---|---|
| `orkun/proto-sweeps` | `qibolab-proto-sweeps` | sweep parameters, parallel groups, functional envelope (duration sweeps, sample rounding) |
| `orkun/proto-pulses` | `qibolab-proto-pulses` | flux pulses, VirtualZ, chirp, flux filters, SHFQC |
| `orkun/proto-acquisition` | `qibolab-proto-acquisition` | acquisition types, averaging, kernels, readout layout in time, fast reset |

Merge order: sweeps, then pulses, then acquisition.
New tests go in `tests/instruments/zi/proto/`, one file per branch (`test_sweeps.py`, `test_pulses.py`, `test_acquisition.py`), so the branches do not touch the same test file.

## Shared interface

`zi/sweep.py` replaces `Amplitudes` with:

```python
Swept = dict[tuple[PulseId | ChannelId, Parameter], SweepParameter]
```

`pulse.play` reads every swept value as `swept.get((op.id, Parameter.x), fixed_value)`.
Until sweeps is merged, the pulses and acquisition branches keep the `amplitudes` argument and read only amplitudes; the merge renames it.

## Per item

LabOne Q names below exist in both versions. Probe results: OK = compiles and runs in emulation.

### orkun/proto-sweeps

| item | construct | probe |
|---|---|---|
| F03 / G1: amplitude sweep given `Readout.probe` | Look up by `probe.id` as well as by the readout id. Probe amplitude stays near time: an SHFQA generator refuses a real-time amplitude sweep ("Too many unique pulses on signal"). | real-time probe amplitude: refused by LabOne Q |
| F04, F05: amplitude sweep on drive and flux | `Experiment.play(..., amplitude=SweepParameter)` inside the real-time loop | OK (drive, HDAWG flux) |
| F06, F07, X23: duration sweep of a pulse and of a `Delay` | Pulse: `play(..., length=SweepParameter)` with a functional pulse (below). Delay: `Experiment.delay(signal, time=SweepParameter)`, real time. A delay on an acquisition channel also delays its probe (as today). | OK |
| F08: duration sweep on flux | Same as F06, on the HDAWG signal. Needs flux pulses (pulses branch). The sweep code is the same as F06; test after merge. | OK (`const` on HDAWG) |
| F09: `duration_interpolated` | Same as F06. LabOne Q samples the functional pulse at each length at compile time. | OK |
| F10: relative phase sweep | `play(..., phase=SweepParameter)`, real time | OK |
| F11: VirtualZ phase sweep | `play(signal, pulse=None, increment_oscillator_phase=SweepParameter)`, real time. Needs VirtualZ play (pulses branch). | OK |
| F12: offset sweep on flux | `SignalCalibration.voltage_offset = SweepParameter`, near time only (a node setting). Values are absolute, as in qm (`set_dc_offset(variable)`). | OK |
| F13 / G2 / X20: near time and real time in one parallel group | Run the whole group in near time. A hardware-modulated oscillator frequency can be swept in near time too. | OK (drive and probe frequency in one near-time sweep) |
| X06: frequency sweep on a DC channel | Clear `ValueError` naming the channel, instead of `AttributeError`. | n/a |
| G6 / X05: durations not a whole number of samples | Play every non-probe pulse as a functional pulse with `length=duration * 1e-9`. LabOne Q then rounds pulses and delays on the same grid. Probe pulses: keep `sampled_pulse` unless the functional gives one waveform per probe signal (check). | OK (`length=41.3 ns`) |

Functional pulse, registered once at module level (a lambda or a local function fails: 26.4 cannot pickle it, 26.10 accepts only plain values in `pulse_parameters`):

```python
@pulse_library.register_pulse_functional
def qibolab_envelope(x, envelope=None, **_):
    env = TypeAdapter(Envelope).validate_python(envelope)
    return env.i(len(x)) + 1j * env.q(len(x))

play(signal, qibolab_envelope(length=..., amplitude=1.0), length=swept_or_fixed,
     pulse_parameters={"envelope": op.envelope.model_dump(mode="json")})
```

Probe: a length sweep of 40, 80, 120 ns compiles to waveforms of 80, 160, 240 samples (26.4.0; 26.10.1 runs, its compiled object has no `waves` attribute to inspect).
`pulse_library.sampled_pulse` with `length=` also compiles, but what it plays is not documented; do not use it for duration sweeps.

Real time vs near time after this branch: frequency of a hardware-modulated channel, amplitude of drive and flux pulses, duration, relative phase and VirtualZ phase run in real time. Probe frequency, probe amplitude and DC offset run in near time.

### orkun/proto-pulses

| item | construct | probe |
|---|---|---|
| F28, F35, F36, F37 on flux, X09 coupler pulses | `play` on the HDAWG signal of a `DcChannel` (real samples, no oscillator). Same `waveform()` path as drive. | OK (sampled exponential, amplitude sweep) |
| F27: VirtualZ | `play(signal, pulse=None, increment_oscillator_phase=op.phase)`. Sign: +phase, as qm (`frame_rotation_2pi`) and qblox (`set_ph_delta`). Only on hardware-modulated IQ channels; refuse elsewhere. | OK |
| X08: CZ (flux pulse + VirtualZ) | Follows from F27 and F28. | follows |
| G5 / X02: chirp | Constant rate `(rate, units)` only: multiply the samples by `exp(1j * pi * rate_Hz_per_ns * 1e-9 * t_ns**2)`, frequency starting at the oscillator frequency, as QUA. Units: the 9 QUA units in qm `program/arguments.py`. Piecewise chirps (list forms): refuse. | OK (sampled) |
| F39: flux filters | `SignalCalibration.precompensation = Precompensation(exponential=[ExponentialCompensation(timeconstant, amplitude)], FIR=FIRCompensation(coefficients))`. qibolab `ExponentialFilter.tau` is in samples: `timeconstant = tau / sampling_rate * 1e-9`. FIR coefficients as given. Limits to check in the HDAWG manual: 8 exponentials, 40 FIR taps. Needs the HDAWG `PC` option in `ZiInstrument.options`; without it LabOne Q refuses ("Precompensation is not supported"). | OK with `HDAWG8/PC` |
| G7 / X17: SHFQC | `laboneq.dsl.device.instruments.SHFQC`; ports `SGCHANNELS/{n}/OUTPUT`, `QACHANNELS/0/OUTPUT`, `QACHANNELS/0/INPUT`. Add `"SHFQC"` to `InstrumentKind`, `_INSTRUMENT_CLASSES`, `_PORTS`, `_MAX_PORTS`. Oscillator: software on the QA part, hardware on the SG part (choose by port type, not by instrument kind). | OK (`SHFQC/QC6CH` with HDAWG and PQSC) |
| UHFQA, HDAWG IQ outputs, SHFPPC | Not in this round. Keep the refusal, with a message naming the instrument. SHFPPC would use `SignalCalibration.amplifier_pump`. | n/a |
| X07: pulse on an acquisition channel | Clear refusal. | n/a |

### orkun/proto-acquisition

| item | construct | probe |
|---|---|---|
| F17: DISCRIMINATION | `acquire_loop_rt(acquisition_type=AcquisitionType.DISCRIMINATION)`, `SignalCalibration.threshold = config.threshold` (one float for 2 states; a list is refused for 2 states). Rotation by `iq_angle`: kernel times `exp(1j * iq_angle)` (as the earlier qibolab zhinst driver). Result: real part, shape `bins` (no i/q axis). | OK; emulation returns a constant (42), so the state mapping is unknown |
| F18: RAW | `AcquisitionType.RAW`, `acquire(signal, handle, length=...)` without kernel. Result shape: `bins + (samples, 2)`. LabOne Q allows one RAW acquire per averaging loop (docstring); refuse more than that. Up to 4096 samples. | OK: cyclic (2000,), single shot (4, 2000), with sweep (3, 2000) |
| F21: SEQUENTIAL | `AveragingMode.SEQUENTIAL` in `_AVERAGING`. Same result axes as CYCLIC. | OK |
| X01: default options | Follows from F17 (default is DISCRIMINATION + SINGLESHOT). | follows |
| F41: integration kernel | `acquire(..., kernel=pulse_library.sampled_pulse(config.kernel))` (complex). Length limit: 4096 samples (2.048 us at 2 GSa/s) for INTEGRATION. | OK |
| X11: `smearing` | Not used yet; document. | n/a |
| G3 / X03, F47, F25: readouts at different times, two readouts of one qubit, separate probe and Acquisition | Change the layout of one sequence: one subsection per run of ops of one channel, and one subsection per readout (probe play + acquire). LabOne Q does not allow ops and sections mixed in one section. Readouts on one QA line that overlap in time must start and end together; otherwise refuse before compiling with a clear message. A separate `Acquisition` pairs with the probe `Pulse` played at the same time on the probe channel. | Same start: OK. Different start, no overlap: OK. Mid-circuit (two readouts of one qubit): OK. Overlap with different start: refused by LabOne Q |
| G4 / X04: two probe waveforms or amplitudes on one probe signal in one real-time loop | SHFQA limit. Refuse before compiling, naming the qubit and the two pulses. | refused by LabOne Q in both versions |
| F45: fast reset | After each readout: `with match(handle=...)`, `case(state=0)` empty, `case(state=1)` plays the reset pulse on the qubit drive. Works with INTEGRATION and DISCRIMINATION (threshold set). The controller does not know the natives: add a `ZiController` field mapping each acquisition channel to `(drive channel, pulse)`; refuse fast reset for a channel without it. | OK (emulation; feedback latency not modelled) |

### acquisition: done and not done

Done on `orkun/proto-acquisition` (emulation only): F21 SEQUENTIAL, F17 DISCRIMINATION with X01, F18 RAW, F45 fast reset, G7 SHFQC.
SHFQC was listed under `orkun/proto-pulses`; it is done here, so proto-pulses should drop it.

Implemented, but the values depend on hardware facts not checked:
- DISCRIMINATION: kernel `-exp(1j * iq_angle)` times unit weights, threshold `-threshold`. Reason: LabOne Q sets the 2-state assignment vector so that above the threshold is state 0 (`controller/devices/qachannel.py`, `_calc_theoretical_assignment_vec`), while qibolab (qm, qibocal) has state 1 when `Re((I + 1j Q) exp(1j iq_angle)) > threshold`. This assumes the integration is linear in the kernel (convention of `laboneq.analysis.calculate_integration_kernels`). Emulation returns placeholder states, so the state mapping and the sign of the angle need hardware data.
- RAW: one acquisition per call (LabOne Q `AcquisitionType.RAW` docstring; the emulator accepts more). Traces are demodulated in software at the configured probe frequency relative to the LO, as qblox and qm; with a probe frequency sweep the configured value is used. The sign of the SHFQA scope samples is not checked. SEQUENTIAL with RAW and traces over 4096 samples are refused by LabOne Q.
- Fast reset: needs DISCRIMINATION; reset pulses come from the new field `ZiController.reset_pulses` (acquisition channel to (channel, pulse)). Feedback latency is not modelled in emulation.

Not done in this round (not requested): F41 kernels, readout layout (G3/X03, F47, F25), G4 refusal.

## Version notes

- `pulse_library.sampled_pulse_complex` and `sampled_pulse_real` are gone in 26.10.1. Use `sampled_pulse` (both versions).
- 26.10.1 instrument classes have no `device_class` argument; `create_connection` takes keywords only after `ports`. The driver already passes keywords.
- Baseline on e8757e58: 72 passed on 26.4.0; 70 passed and 2 failed on 26.10.1 (`test_calibration.py::test_compile`, `test_controller.py::test_disconnect_zeroes_dc_offsets`: the tests read `scheduled_experiment.recipe`, absent in 26.10.1). These 2 are not regressions.

## Driver tests that assert a refusal

`tests/instruments/zi/test_play.py::test_unsupported` (duration sweep, mixed timing, fast reset, SEQUENTIAL, DISCRIMINATION, RAW, default options), `test_calibration.py::test_flux_filters_unsupported`, `test_controller.py::test_unsupported_instrument` (SHFQC).
Each branch updates only the cases of the features it implements, and says so in the commit message.

## Working rules

- Venvs per worktree: `.venv` from `uv sync --all-extras --all-groups` (laboneq 26.4.0); `.venv-2610` the same with `UV_PROJECT_ENVIRONMENT=.venv-2610`, then `uv pip install --python .venv-2610/bin/python laboneq==26.10.1 zhinst-core==26.10.0.988`.
- Run probes from a scratch directory: LabOne Q writes `laboneq_output/` into the current directory, and it is not in `.gitignore`.
- Run `pytest tests/instruments/zi` on both venvs before each commit.
