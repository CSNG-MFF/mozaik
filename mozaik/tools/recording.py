"""Efficient MPI gathering for the regular recordings produced by Mozaik."""

import neo
import numpy
import quantities as pq

import mozaik


SPIKE_ANNOTATIONS = {"channel_id", "source_index", "source_population"}
SIGNAL_ANNOTATIONS = {"channel_ids", "source_population"}


def gather_recording(block):
    """Gather a regular one-segment recording, with a generic safe fallback."""
    comm = mozaik.mpi_comm
    if comm is None or comm.size == 1:
        return block

    supported = comm.allreduce(int(_is_supported(block))) == comm.size
    if not supported:
        from pyNN.recording import gather_blocks

        return gather_blocks(block)

    packets = comm.gather(_pack(block), root=mozaik.MPI_ROOT)
    if comm.rank != mozaik.MPI_ROOT:
        return block

    return _merge(block, packets)


def _is_supported(block):
    if len(block.segments) != 1 or block.groups:
        return False
    segment = block.segments[0]
    known_objects = len(segment.spiketrains) + len(segment.analogsignals)
    return (
        len(segment.data_children) == known_objects
        and all(
            set(train.annotations) == SPIKE_ANNOTATIONS
            and train.name is None
            and train.description is None
            and train.file_origin is None
            for train in segment.spiketrains
        )
        and len({signal.name for signal in segment.analogsignals})
        == len(segment.analogsignals)
        and all(
            set(signal.annotations) == SIGNAL_ANNOTATIONS
            and set(signal.array_annotations) == {"channel_index"}
            for signal in segment.analogsignals
        )
    )


def _pack(block):
    segment = block.segments[0]
    trains = list(segment.spiketrains)
    spikes = None
    if trains:
        spikes = {
            "times": numpy.concatenate([train.magnitude for train in trains]),
            "counts": numpy.array([train.size for train in trains]),
            "channel_ids": numpy.array([train.annotations["channel_id"] for train in trains]),
            "source_indices": numpy.array([train.annotations["source_index"] for train in trains]),
            "source_population": trains[0].annotations["source_population"],
            "t_start": numpy.array([train.t_start.magnitude for train in trains]),
            "t_stop": numpy.array([train.t_stop.magnitude for train in trains]),
            "units": str(trains[0].dimensionality),
            "time_units": str(trains[0].t_start.dimensionality),
        }

    signals = []
    for signal in segment.analogsignals:
        signals.append(
            {
                "name": signal.name,
                "values": signal.magnitude,
                "units": str(signal.dimensionality),
                "t_start": signal.t_start.magnitude,
                "time_units": str(signal.t_start.dimensionality),
                "sampling_period": signal.sampling_period.magnitude,
                "sampling_units": str(signal.sampling_period.dimensionality),
                "channel_ids": signal.annotations["channel_ids"],
                "channel_index": signal.array_annotations["channel_index"],
                "description": signal.description,
                "dtype": signal.dtype.str,
                "file_origin": signal.file_origin,
                "source_population": signal.annotations["source_population"],
            }
        )
    return {"spikes": spikes, "signals": signals}


def _merge(root_block, packets):
    segment = root_block.segments[0]
    trains = list(segment.spiketrains)
    for packet in packets[1:]:
        spikes = packet["spikes"]
        if spikes is None:
            continue
        offset = 0
        for i, count in enumerate(spikes["counts"]):
            count = int(count)
            train = neo.SpikeTrain(
                spikes["times"][offset : offset + count],
                units=spikes["units"],
                t_start=pq.Quantity(spikes["t_start"][i], spikes["time_units"]),
                t_stop=pq.Quantity(spikes["t_stop"][i], spikes["time_units"]),
                channel_id=int(spikes["channel_ids"][i]),
                source_index=int(spikes["source_indices"][i]),
                source_population=spikes["source_population"],
            )
            train.segment = segment
            trains.append(train)
            offset += count
    segment.spiketrains = sorted(trains, key=lambda train: train.annotations["channel_id"])

    by_name = {}
    for packet in packets:
        for signal in packet["signals"]:
            by_name.setdefault(signal["name"], []).append(signal)
    segment.analogsignals = [_merge_signal(signals, segment) for signals in by_name.values()]
    return root_block


def _merge_signal(signals, segment):
    first = signals[0]
    signal = neo.AnalogSignal(
        numpy.hstack([item["values"] for item in signals]),
        units=first["units"],
        dtype=numpy.dtype(first["dtype"]),
        t_start=pq.Quantity(first["t_start"], first["time_units"]),
        sampling_period=pq.Quantity(first["sampling_period"], first["sampling_units"]),
        name=first["name"],
        description=first["description"],
        file_origin=first["file_origin"],
        channel_ids=numpy.concatenate([item["channel_ids"] for item in signals]),
        source_population=first["source_population"],
        array_annotations={
            "channel_index": numpy.concatenate([item["channel_index"] for item in signals])
        },
    )
    signal.segment = segment
    return signal
