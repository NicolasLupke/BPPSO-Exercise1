import math
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only needed for type checkers
    from pm4py.objects.petri_net.obj import PetriNet  # type: ignore[import]


def _ensure_petri_net(petri_net: Any) -> Any:
    """
    Minimal runtime validation that what we received behaves like a PM4Py Petri net.
    """
    if petri_net is None:
        raise ValueError("A Petri net instance is required to compute simplicity metrics.")
    for attr in ("places", "transitions"):
        if not hasattr(petri_net, attr):
            raise TypeError(
                f"The provided object does not expose the expected '{attr}' attribute."
            )
    if not hasattr(petri_net, "arcs"):
        raise TypeError("The provided object does not expose the expected 'arcs' attribute.")
    return petri_net


def _outgoing_transitions(place: Any) -> Iterable[Any]:
    for arc in getattr(place, "out_arcs", []):
        target = getattr(arc, "target", None)
        if target is not None and hasattr(target, "in_arcs"):
            yield target


def entropy_simplicity_metric(petri_net: Any) -> float:
    """
    Entropy-based simplicity metric in the range [0, 1].

    A place that branches uniformly over `n` transitions contributes an entropy of log(n).
    We normalise per place and return 1 minus the average normalised entropy, so
    deterministic behaviour (no branching) scores 1.0.
    """
    net = _ensure_petri_net(petri_net)

    if not net.places:
        return 1.0

    normalised_entropies = []
    for place in net.places:
        transitions = list(_outgoing_transitions(place))
        branching = len(transitions)
        if branching <= 1:
            continue

        probability = 1.0 / branching
        entropy = -branching * probability * math.log(probability)
        max_entropy = math.log(branching)
        if max_entropy > 0:
            normalised_entropies.append(entropy / max_entropy)

    if not normalised_entropies:
        return 1.0

    mean_entropy = sum(normalised_entropies) / len(normalised_entropies)
    score = 1.0 - mean_entropy
    return max(0.0, min(1.0, score))


def size_simplicity_metric(petri_net: Any) -> int:
    """
    Simple size metric counting the structural elements of the net.
    Larger values indicate a less simple model.
    """
    net = _ensure_petri_net(petri_net)
    return len(net.places) + len(net.transitions) + len(net.arcs)

