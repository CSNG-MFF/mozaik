from types import SimpleNamespace

import neo
import numpy
import quantities as pq

import mozaik
from mozaik.tools.recording import _is_supported, _pack, gather_recording


class RootComm:
    rank = 0
    size = 2

    def __init__(self, other_packet=None, supported=True):
        self.other_packet = other_packet
        self.supported = supported

    def allreduce(self, value):
        return self.size if self.supported and value else 0

    def gather(self, value, root):
        assert root == 0
        return [value, self.other_packet]


def make_block(channel_id, value):
    block = neo.Block()
    segment = neo.Segment()
    block.segments.append(segment)
    segment.block = block

    train = neo.SpikeTrain([value] * pq.ms, t_stop=10 * pq.ms)
    train.annotate(
        channel_id=channel_id,
        source_index=channel_id,
        source_population="population",
    )
    train.segment = segment
    segment.spiketrains.append(train)

    signal = neo.AnalogSignal(
        numpy.full((2, 1), value),
        units=pq.mV,
        dtype=numpy.float32,
        sampling_period=1 * pq.ms,
        name="v",
        description="membrane potential",
        file_origin="NEST",
        channel_ids=numpy.array([channel_id]),
        source_population="population",
        array_annotations={"channel_index": numpy.array([channel_id])},
    )
    signal.segment = segment
    segment.analogsignals.append(signal)
    return block


def test_gather_regular_recording(monkeypatch):
    root = make_block(2, 2.0)
    other = make_block(1, 1.0)
    monkeypatch.setattr(mozaik, "mpi_comm", RootComm(_pack(other)))

    merged = gather_recording(root)

    segment = merged.segments[0]
    assert [train.annotations["channel_id"] for train in segment.spiketrains] == [1, 2]
    numpy.testing.assert_array_equal(segment.analogsignals[0].magnitude, [[2, 1], [2, 1]])
    numpy.testing.assert_array_equal(
        segment.analogsignals[0].array_annotations["channel_index"], [2, 1]
    )
    assert segment.analogsignals[0].dtype == numpy.dtype("float32")
    assert segment.analogsignals[0].description == "membrane potential"
    assert segment.analogsignals[0].file_origin == "NEST"


def test_unsupported_recording_uses_generic_gather(monkeypatch):
    block = make_block(1, 1.0)
    block.segments.append(neo.Segment())
    sentinel = object()
    monkeypatch.setattr(mozaik, "mpi_comm", RootComm(supported=False))
    monkeypatch.setattr(
        "pyNN.recording.gather_blocks",
        lambda candidate: sentinel if candidate is block else None,
    )

    assert not _is_supported(block)
    assert gather_recording(block) is sentinel


def test_serial_recording_is_unchanged(monkeypatch):
    block = make_block(1, 1.0)
    monkeypatch.setattr(mozaik, "mpi_comm", SimpleNamespace(size=1))

    assert gather_recording(block) is block
